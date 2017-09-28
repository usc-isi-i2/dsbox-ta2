import os
import json
import copy
import pandas as pd

from dsbox.schema.data_profile import DataProfile
from dsbox.schema.dataset_schema import VariableFileType
from dsbox.schema.profile_schema import DataProfileType as dpt

DEFAULT_DATA_SCHEMA = "dataSchema.json"
DEFAULT_TRAINING_DATA = "trainData.csv.gz"
DEFAULT_TRAINING_TARGETS = "trainTargets.csv.gz"
DEFAULT_TEST_DATA = "testData.csv.gz"
DEFAULT_RAW_DATA_DIR = "raw_data"

class DataPackage(object):
    '''
    A Data Package (schema + csv files + raw data)
    '''
    def __init__(self):
        self.schema = None # The schema file
        self.data_directory = None # The data directory

        self.input_columns = []
        self.target_columns = []
        self.input_data = pd.DataFrame()
        self.target_data = pd.DataFrame()

        self.index_column = None
        self.boundary_columns = []
        self.media_type = None
        self.nested_table = dict()

    """
    Load the data package given a schema, a data directory, column filters
    """
    def load_data(self, schema, data_directory, input_filters=None, target_filters=None, test_data=False):
        self.schema = schema # The schema file

        # Get column information
        schema_data = self.load_json(self.schema)
        self.input_columns = schema_data['trainData']['trainData']
        if input_filters is not None:
            self.input_columns = self.filter_columns(self.input_columns, input_filters)
        self.target_columns = schema_data['trainData']['trainTargets']
        if target_filters is not None:
            self.target_columns = self.filter_columns(self.target_columns, target_filters)

        # Get special columns
        self.index_column = self.get_index_column(self.input_columns)
        self.boundary_columns = self.get_boundary_columns(self.input_columns)

        # Detect media type
        self.media_type = self.get_media_type(schema_data, self.input_columns)

        # Load data
        self.data_directory = data_directory
        if not test_data:
            self.input_data = self.read_data(data_directory + os.sep + DEFAULT_TRAINING_DATA,
                                                     self.input_columns, self.index_column)
            self.target_data = self.read_data(data_directory + os.sep + DEFAULT_TRAINING_TARGETS,
                                                     self.target_columns, self.index_column, labeldata=True)
        else:
            self.input_data = self.read_data(data_directory + os.sep + DEFAULT_TEST_DATA,
                                                     self.input_columns, self.index_column)

    def filter_columns(self, columns, filters):
        new_columns = []
        for col in columns:
            if (col['varName'] in filters) or ("*" in filters):
                new_columns.append(col)
        return new_columns

    def load_json(self, jsonfile):
        with open(jsonfile) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d

    def read_data(self, csvfile, cols, indexcol, labeldata=False):
        # We look for the .csv.gz file by default. If it doesn't exist, try to load the .csv file
        if not os.path.exists(csvfile) and csvfile.endswith('.gz'):
            csvfile = csvfile[:-3]

        # Read the csv file
        df = pd.read_csv(csvfile)
        raw_data_path = os.path.dirname(csvfile) + os.sep + DEFAULT_RAW_DATA_DIR

        # Filter columns if specified
        if len(cols) > 0:
            colnames = []
            for col in cols:
                colnames.append(col['varName'])

            # Remove columns not specified
            for colname, col in df.iteritems():
                if colname not in colnames:
                    df.drop(colname, axis=1, inplace=True)

            # Check for nested tabular data files, and load them in
            tabular_columns = []
            index_columns = []
            for col in cols:
                colname = col['varName']
                varRole = col.get('varRole', None)
                varType = col.get('varType', None)
                varFileType = col.get('varFileType', None)
                if varRole == 'index':
                    index_columns.append(colname)
                if varType == 'file' and varFileType == 'tabular':
                    tabular_columns.append(colname)
                    filename = df.loc[:, colname].unique()
                    if len(filename) > 1:
                        raise AssertionError('Expecting one unique filename per column: {}'.format(colname))
                    filename = filename[0]
                    if not filename in self.nested_table:
                        csvfile = raw_data_path + os.sep + filename
                        if not os.path.exists(csvfile):
                            csvfile += '.gz'
                        nested_df = self.read_data(csvfile, [], None)
                        self.nested_table[filename] = nested_df


            # Match index columns to tabular columns
            if len(tabular_columns) == len(index_columns) - 1:
                # New r_32 dataset has two tabular columns and three index columns. Need to remove d3mIndex
                # New r_26 dataset has exactly one tabular column and one index column (d3mIndex)
                index_columns = index_columns[1:]
            if not len(tabular_columns) == len(index_columns):
                raise AssertionError('Number tabular and index columns do not match: {} != {}'
                                     .format(len(tabular_columns), (index_columns)))

            # Check all columns for special roles
            for col in cols:
                colname = col['varName']
                varRole = col.get('varRole', None)
                varType = col.get('varType', None)
                varFileType = col.get('varFileType', None)
                fileobjects = {}
                if varRole == 'file' or varType == 'file':
                    # If the role is "file", then load in the raw data files
                    if self.media_type in (VariableFileType.TEXT, VariableFileType.IMAGE, VariableFileType.AUDIO, VariableFileType.GRAPH):
                        for index, row in df.iterrows():
                            filepath = raw_data_path + os.sep + row[colname]
                            if self.media_type == VariableFileType.TEXT:
                                # Plain data load for text files
                                with open(filepath, 'rb') as myfile:
                                    txt = myfile.read()
                                    df.set_value(index, colname, txt)
                            elif self.media_type == VariableFileType.IMAGE:
                                # Load image files using keras with a standard target size
                                # TODO: Make the (224, 224) size configurable
                                from keras.preprocessing import image
                                df.set_value(index, colname, image.load_img(filepath, target_size=(224, 224)))
                            elif self.media_type == VariableFileType.GRAPH:
                                import networkx as nx
                                nxobject = fileobjects.get(filepath, None)
                                if nxobject is None:
                                    nxobject = nx.read_gml(filepath)
                                fileobjects[filepath] = nxobject
                                df.set_value(index, colname, nxobject)
                            elif self.media_type == VariableFileType.AUDIO:
                                # Load audio files
                                import librosa
                                # Load file
                                try:
                                    print (filepath)
                                    start = None
                                    end = None
                                    bcols = self.boundary_columns
                                    kwargs = {'sr':None}
                                    if len(bcols) == 2:
                                        start = float(row[bcols[0]])
                                        end = float(row[bcols[1]])
                                        if start > end:
                                            tmp = start
                                            start = end
                                            end = tmp
                                        kwargs['offset'] = start
                                        kwargs['duration'] = end-start
                                    (audio_clip, sampling_rate) = librosa.load(filepath, **kwargs)
                                    df.set_value(index, colname, (audio_clip, sampling_rate))
                                except Exception as e:
                                    df.set_value(index, colname, None)

            origdf = df
            for file_colname, index_colname in zip(tabular_columns, index_columns):
                # FIXME: Assumption here that all entries for the filename are the same per column
                filename = origdf.iloc[0][file_colname]

                # Merge the nested table with parent table on the index column
                nested_table = self.nested_table[filename]
                df = pd.merge(df, nested_table, on=index_colname)

                # Remove file and index columns since the content has been replaced
                del df[file_colname]
                if index_colname != indexcol:
                    del df[index_colname]
                ncols = []
                for col in cols:
                    if col['varName'] not in [file_colname, index_colname]:
                        ncols.append(col)
                cols = ncols
                # Add nested table columns
                for nested_colname in nested_table.columns:
                    if not nested_colname == index_colname:
                        cols.append({'varName': nested_colname})

            if cols:
                if labeldata:
                    self.target_columns = cols
                else:
                    self.input_columns = cols

        if indexcol is not None:
            # Set the table's index column
            df = df.set_index(indexcol, drop=True)

            # Remove the index column from the list
            for col in cols:
                if col['varName'] == indexcol:
                    cols.remove(col)

            if not labeldata:
                # Check if we need to set the media type for any columns
                profile = DataProfile(df)
                if profile.profile[dpt.TEXT]:
                    if self.media_type == VariableFileType.TABULAR or self.media_type is None:
                        self.media_type = VariableFileType.TEXT
                        for colname in profile.columns.keys():
                            colprofile = profile.columns[colname]
                            if colprofile[dpt.TEXT]:
                                for col in self.input_columns:
                                    if col['varName'] == colname:
                                        col['varType'] = 'file'
                                        col['varFileType'] = 'text'
                                        col['varFileFormat'] = 'text/plain'
        return df


    def get_index_column(self, columns):
        for col in columns:
            if col['varRole'] == 'index':
                return col['varName']
        return None

    def get_boundary_columns(self, columns):
        cols = []
        for col in columns:
            if col.get('varRole', None) == 'boundary':
                cols.append(col['varName'])
        return cols

    def get_media_type(self, schema, cols):
        if schema.get('rawData', False):
            for col in cols:
                varType = col.get('varType', None)
                if varType == 'file':
                    return VariableFileType(col.get('varFileType'))
        return None




class DataManager(object):
    """
    The Manage Data management Class.
    Combines data from several data packages, and provides a API to edit/manage data
    """
    def __init__(self):
        self.original_data = None
        self.data = None # Combined DataPackage
        self.data_parts = {} # Hash of data id to DataPackage parts

    """
    This function creates train_data and train_labels from trainData.csv and trainTargets.csv
    """
    def initialize_training_data_from_defaults(self, schema_file, data_directory):
        print("Loading Data..")
        self.data_parts = {}
        self.data_parts[data_directory] = DataPackage()
        self.data_parts[data_directory].load_data(
            schema_file, data_directory
        )
        self._combine_data_parts()

    """
    This function creates train_data and train_labels from trainData.csv and trainTargets.csv
    """
    def initialize_test_data_from_defaults(self, schema_file, data_directory):
        print("Loading Data..")
        self.data_parts = {}
        self.data_parts[data_directory] = DataPackage()
        self.data_parts[data_directory].load_data(
            schema_file,
            data_directory,
            test_data=True
        )
        self._combine_data_parts()

    """
    This function creates input data, and target data from the set of train and target features
    """
    def initialize_training_data_from_features(self, train_features, target_features):
        data_directories = []
        input_data_features_map = {}
        target_data_features_map = {}
        self.data_parts = {}

        print("Loading Data..")
        for feature in train_features:
            input_data_features = input_data_features_map.get(feature.data_directory, [])
            input_data_features.append(feature.feature_id)
            input_data_features_map[feature.data_directory] = input_data_features
            if feature.data_directory not in data_directories:
                data_directories.append(feature.data_directory)

        for feature in target_features:
            target_data_features = target_data_features_map.get(feature.data_directory, [])
            target_data_features.append(feature.feature_id)
            target_data_features_map[feature.data_directory] = target_data_features
            if feature.data_directory not in data_directories:
                data_directories.append(feature.data_directory)

        for data_directory in data_directories:
            input_features = input_data_features_map.get(data_directory, None)
            target_features = target_data_features_map.get(data_directory, None)
            schema_file = data_directory + os.sep + DEFAULT_DATA_SCHEMA
            self.data_parts[data_directory] = DataPackage()
            self.data_parts[data_directory].load_data(
                schema_file,
                data_directory,
                input_filters=input_features,
                target_filters=target_features
            )
        self._combine_data_parts()

    """
    This function creates input data from the set of test features
    """
    def initialize_test_data_from_features(self, test_features):
        data_directories = []
        input_data_features_map = {}
        self.data_parts = {}

        print("Loading Data..")
        for feature in test_features:
            input_data_features = input_data_features_map.get(feature.data_directory, [])
            input_data_features.append(feature.feature_id)
            input_data_features_map[feature.data_directory] = input_data_features
            if feature.data_directory not in data_directories:
                data_directories.append(feature.data_directory)

        for data_directory in data_directories:
            input_features = input_data_features_map.get(data_directory, None)
            schema_file = data_directory + os.sep + DEFAULT_DATA_SCHEMA
            self.data_parts[data_directory] = DataPackage()
            self.data_parts[data_directory].load_data(
                schema_file,
                data_directory,
                input_filters=input_features,
                test_data=True
            )
        self._combine_data_parts()


    """
    This function combines the individual parts into one
    """
    def _combine_data_parts(self):
        self.data = DataPackage()
        self.data.schema = []
        for partid in self.data_parts.keys():
            dp = self.data_parts[partid]
            self.data.schema.append(dp.schema)
            self.data.input_columns = self.data.input_columns + dp.input_columns
            self.data.input_data = pd.concat([self.data.input_data, dp.input_data], axis=1)
            self.data.target_columns = self.data.target_columns + dp.target_columns
            self.data.target_data = pd.concat([self.data.target_data, dp.target_data], axis=1)
            self.data.boundary_columns = self.data.boundary_columns + dp.boundary_columns
            if dp.index_column is not None:
                self.data.index_column = dp.index_column
            if dp.media_type is not None:
                self.data.media_type = dp.media_type
        # Take a copy of the original data
        self.original_data = copy.deepcopy(self.data)
