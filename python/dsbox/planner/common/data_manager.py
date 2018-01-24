import os
import sys
import json
import copy
import inspect
import warnings
import importlib
import numpy as np
import pandas as pd

import multiprocessing
from multiprocessing import Pool

from dsbox.schema.dataset_schema import VariableFileType
#from dsbox.schema.data_profile import DataProfile
#from dsbox.schema.profile_schema import DataProfileType as dpt

DATASET_SCHEMA_VERSION = '3.0'
DEFAULT_DATA_DOC = "datasetDoc.json"

class DataManager(object):
    input_data = None
    input_columns = None
    index_column = None
    target_data = None
    target_columns = None
    media_type = None

    """
    The Manage Data management Class.
    Combines data from several datasets, and provides a API to edit/manage data
    """

    """
    This function creates 2 dataframes:
    - Training/Testing data
    - Target labels
    """
    def initialize_data(self, problem, datasets, view=None):
        print("Loading Data..")
        sys.stdout.flush()

        splits_df = self._get_datasplits(problem, view)

        dsmap = {}
        for dataset in datasets:
            dsmap[dataset.dsID] = dataset

        dataframes = {}
        # Split dataset into data and targets
        for dsid, targets in problem.dataset_targets.items():
            if dsid in dsmap:
                dataset = dsmap[dsid]
                dataframes[dsid] = {}
                restargets = {}
                for target in targets:
                    tresid = target["resID"]
                    if tresid not in restargets:
                        restargets[tresid] = []
                    restargets[tresid].append(target)
                for resid, targets in restargets.items():
                    if resid in dataset.resources:
                        resource = dataset.resources[resid]
                        if type(resource) is TableResource:
                            # Select appropriate rows of the resource
                            if splits_df is not None:
                                resource.df = resource.df.loc[splits_df.index]
                                resource.split = True
                            # Select targets
                            target_cols = list(map(lambda x: x['colName'], targets))
                            targetdf = copy.copy(resource.df[target_cols])
                            # Drop target columns from df
                            resource.df.drop(target_cols, axis=1, inplace=True, errors='ignore')
                            dataframes[dsid][resid] = {"target_cols": target_cols, "targets": targetdf}

        # Filter dataset by filters (if any)
        for dsid, filters in problem.dataset_filters.items():
            if dsid in dsmap:
                dataset = dsmap[dsid]
                if dsid not in dataframes:
                    dataframes[dsid] = {}
                resfilters = {}
                if len(filters) > 0:
                    for filt in filters:
                        filtresid = filt["resID"]
                        if filtresid not in resfilters:
                            resfilters[filtresid] = []
                        resfilters[filtresid].append(filt)
                    for resid, filters in resfilters.items():
                        if resid in dataset.resources:
                            resource = dataset.resources[resid]
                            if type(resource) is TableResource:
                                if splits_df is not None and not resource.split:
                                    resource.df = resource.df.loc[splits_df.index]
                                    resource.split = True
                                filter_cols = list(map(lambda x: x['colName'], filters))
                                if resource.index_column in filter_cols:
                                    filter_cols.remove(resource.index_column)
                                resource.df = copy.copy(resource.df[filter_cols])
                                if resid not in dataframes[dsid]:
                                    dataframes[dsid][resid] = {}
                                dataframes[dsid][resid]["filter_cols"] = filter_cols
                '''
                else:
                    for resid, resource in dataset.resources.items():
                        if type(resource) is TableResource:
                            if splits_df is not None and not resource.split:
                                resource.df = resource.df.loc[splits_df.index]
                                resource.split = True
                            resource.df = copy.copy(resource.df)
                            if resid not in dataframes[dsid]:
                                dataframes[dsid][resid] = {}
                '''

        self.input_data = pd.DataFrame()
        self.target_data = pd.DataFrame()
        self.input_columns = []
        self.target_columns = []

        # Resolve references
        for dsid, resdfs in dataframes.items():
            dsmap[dsid].resolve_references()

        # Combine multiple dataframes
        for dsid, resdfs in dataframes.items():
            if dsmap[dsid].resType is not None and self.media_type is None:
                self.media_type = VariableFileType(dsmap[dsid].resType)
            for resid, resdf in resdfs.items():
                resource = dsmap[dsid].resources[resid]
                for col in resource.columns:
                    if ("target_cols" in resdf and
                            col['colName'] in resdf["target_cols"]):
                        self.target_columns.append(col)
                    elif ("filter_cols" not in resdf or
                            col['colName'] in resdf['filter_cols']):
                        if "index" in col['role']:
                            self.index_column = col['colName']
                        else:
                            self.input_columns.append(col)

                    # Bit of a hack: Check if the target resource has any string columns
                    # Mark media as text if so
                    if ("target_cols" in resdf and
                            col['colType'] == "string" and
                            self.media_type is  None):
                        self.media_type = VariableFileType("text")

                self.input_data = pd.concat([self.input_data, resource.df])
                if "targets" in resdf:
                    self.target_data = pd.concat([self.target_data, resdf["targets"]])
                self.input_data.columns = list(map(lambda x: x['colName'], self.input_columns))
                self.target_data.columns = list(map(lambda x: x['colName'], self.target_columns))


    def _get_datasplits(self, problem, view=None):
        """
        Returns the data splits in a dataframe
        """
        if problem.splits_file is None:
            return None
        df = pd.read_csv(problem.splits_file, index_col='d3mIndex')
        if view is None:
            return df
        elif view.upper() == 'TRAIN':
            df = df[df['type'] == 'TRAIN']
            return df
        elif view.upper() == 'TEST':
            df = df[df['type'] == 'TEST']
            return df

    def _get_target_columns(self, df, targets):
        target_cols = []
        colIndices = {}
        for i in range(0, len(df.columns)):
            colIndices[str(df.columns[i])] = i
        for target in targets:
            target_colname = target['colName']
            assert(target_colname in colIndices)
            target_cols.append(colIndices[target_colname])
        return target_cols

class Dataset(object):
    """
    The Dataset class
    It contains a list of data resources
    """
    dsHome = None
    dsDoc = None
    dsID = None
    about = None
    resources = {}
    default_resource = None
    resType = None

    def load_dataset(self, datasetPath, datasetDoc=None):
        self.dsHome = datasetPath

        # read the schema in dsHome
        if datasetDoc is None:
            datasetDoc = os.path.join(self.dsHome, DEFAULT_DATA_DOC)

        assert os.path.exists(datasetDoc)
        with open(datasetDoc, 'r') as f:
            self.dsDoc = json.load(f)

        # make sure the versions line up
        self.about = self.dsDoc["about"]
        if self.about['datasetSchemaVersion'] != DATASET_SCHEMA_VERSION:
            warnings.warn("Dataset Schema version mismatch")

        self.dsID = self.about["datasetID"]
        for res in self.dsDoc["dataResources"]:
            resource = DataResource.initialize(res, self.dsHome)
            if resource.resType != "table":
                self.resType = resource.resType
            self.resources[resource.resID] = resource

        self.load_resources()

    def load_resources(self):
        # Load all resources
        for resid, res in self.resources.items():
            res.load()
            if type(res) is TableResource:
                if res.resPath.endswith("learningData.csv"):
                    self.default_resource = res
        #self.resolve_references()

    # This is called by the data manager after splicing into training/test
    # for efficiency
    def resolve_references(self):
        # Get all references
        references = {}
        for resid, resource in self.resources.items():
            if type(resource) is TableResource:
                for col in resource.columns:
                    referobj = col.get("refersTo", None)
                    if referobj is not None:
                        toresid = referobj["resID"]
                        tores = self.resources.get(toresid, None)
                        if tores is not None:
                            toref = referobj["resObject"]
                            reference = Reference(resource, col, tores, toref)
                            if resid not in references:
                                references[resid] = {}
                            references[resid][col["colName"]] = reference
        for resid, refmap in references.items():
            for colname, reference in refmap.items():
                self.load_reference(reference, references)

    def load_reference(self, reference, references):
        reference.scanned = True
        tores = reference.to_resource
        if type(tores) is TableResource:
            # Check for further reference from this resource, if exists, load that first
            tocol = reference.to_reference["columnName"]
            if tores.resID in references:
                # Load all references from this table
                for col, further_ref in references[tores.resID]:
                    if not further_ref.scanned:
                        self.load_reference(further_ref, references)
        tores.join_with(reference.from_resource, reference)


class DataResource(object):
    """
    The resource contains the actual data. It could of various types
    """
    def __init__(self, resID, resPath, resType, resFormat):
        self.resID = resID
        self.resPath = resPath
        self.resType = resType
        self.resFormat = resFormat
        self.split = False

    '''
    Initial resource loading (if any)
    '''
    def load(self):
        pass

    '''
    Modify the input resource with data from itself based on the reference.
    The input resource has to be a tabular resource
    '''
    def join_with(self, resource, reference):
        assert(type(resource) is TableResource)

    @staticmethod
    def initialize(obj, dsHome):
        resID = obj["resID"]
        resPath = os.path.join(dsHome, obj["resPath"])
        resType = obj["resType"]
        resFormat = obj["resFormat"]
        if resType == "table":
            if "isCollection" in obj and obj["isCollection"]:
                raise NotImplementedError("Resource type 'table isCollection=true' not recognized")
            else:
                columns = obj["columns"]
                return TableResource(resID, resPath, resType, resFormat, columns)
        elif resType == "text":
            return TextResource(resID, resPath, resType, resFormat)
        elif resType == "image":
            return ImageResource(resID, resPath, resType, resFormat)
        elif resType == "audio":
            return AudioResource(resID, resPath, resType, resFormat)
        elif resType == "graph":
            return GraphResource(resID, resPath, resType, resFormat)
        elif resType == "timeseries": 
            if "isCollection" in obj and not obj["isCollection"]:
                raise NotImplementedError("Resource type 'timeseries isCollection=false' not recognized")
            else:
                return TimeSeriesResource(resID, resPath, resType, resFormat)
        else:
            raise NotImplementedError("Resource type '{}' not recognized".format(resType))

class TableResource(DataResource):
    """
    This contains tabular data (csv)
    """
    df = None
    orig_df = None
    columns = []
    index_column = None

    def __init__(self, resID, resPath, resType, resFormat, columns):
        super(TableResource, self).__init__(resID, resPath, resType, resFormat)
        self.initialize_columns(columns)

    def initialize_columns(self, columns):
        index_hash = {}
        name_hash = {}
        for col in columns:
            name_hash[col["colName"]] = col
            index_hash[col["colIndex"]] = col
            #self.columns_list.append(col)
            if "index" in col["role"]:
                self.index_column = col["colName"]
        self.columns = [index_hash[i] for i in sorted(index_hash.keys())]
        self.named_columns = name_hash

    def load(self):
        self.df = pd.read_csv(self.resPath, index_col=self.index_column)
        self.orig_df = copy.copy(self.df)

    def join_with(self, resource, reference):
        assert(type(resource) is TableResource)
        #print ("Joining {} with {}".format(resource.resPath, self.resPath))
        leftcol = reference.from_column["colName"]
        rightcol = reference.to_reference["columnName"]

        # Update columns
        self.update_columns(resource, leftcol, self, rightcol)

        # Update dataframe itself
        leftindex = False
        rightindex = False
        if leftcol == resource.index_column:
            leftcol = None
            leftindex = True
        if rightcol == self.index_column:
            rightcol = None
            rightindex = True
        resource.df = resource.df.merge(
            self.df, how="left",
            left_on=leftcol,
            right_on=rightcol,
            left_index = leftindex,
            right_index = rightindex
        )
        if leftcol is not None:
            del resource.df[leftcol]
        if rightcol is not None:
            del resource.df[rightcol]

    def get_boundary_columns(self):
        boundary_columns = []
        for col in self.columns:
            if "boundaryIndicator" in col["role"]:
                boundary_columns.append(col["colName"])
        return boundary_columns

    def get_column_values(self, row, columns):
        values = []
        for col in columns:
            values.append(row[col])
        return values

    def update_columns(self, resource, rescol, refres, refcol):
        columns = []
        index = 0
        for col in resource.columns:
            if col['colName'] == rescol:
                for newcol in refres.columns:
                    if newcol['colName'] != refcol:
                        newcol['colIndex'] = index
                        columns.append(newcol)
                        index += 1
            else:
                col['colIndex'] = index
                columns.append(col)
                index += 1
        resource.initialize_columns(columns)


class RawResource(DataResource):
    """
    This contains a collection of raw files like images, audio, text, etc
    """
    LOADING_POOL = None

    def __init__(self, resID, resPath, resType, resFormat, numcpus=0):
        if RawResource.LOADING_POOL is None:
            if numcpus == 0:
                numcpus = multiprocessing.cpu_count()
            RawResource.LOADING_POOL = Pool(numcpus)
        super(RawResource, self).__init__(resID, resPath, resType, resFormat)

    def load(self):
        # Preload all images ?
        pass

    '''
    Load a particular item from the resource collection
    '''
    def load_resource(self, filepath, boundary_values):
        pass

    '''
    Unserialize resource after pickling (due to multiprocessing)
    '''
    def unserialize_resource(self, value):
        return value

    def join_with(self, resource, reference):
        assert(type(resource) is TableResource)
        colname = reference.from_column["colName"]
        boundary_columns = resource.get_boundary_columns()
        loadrefs = []
        for index, row in resource.df.iterrows():
            filename = row[colname]
            if type(filename) == float:
                print(self.resPath, filename, type(filename))
            filepath = os.path.join(self.resPath, filename)
            boundary_values = resource.get_column_values(row, boundary_columns)
            ref = RawResource.LOADING_POOL.apply_async(self.load_resource, args=(filepath, boundary_values))
            loadrefs.append({"ref":ref, "index":index, "filename":filename})

        resource.df[colname] = resource.df[colname].astype(object)
        for loadref in loadrefs:
            value = loadref["ref"].get()
            print("Loading .. {}".format(loadref["filename"]))
            sys.stdout.flush()
            resource.df.at[loadref["index"], colname] = self.unserialize_resource(value)

    '''
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['loading_pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
    '''

class TextResource(RawResource):
    """
    A collection of text files
    """
    def __init__(self, resID, resPath, resType, resFormat):
        super(TextResource, self).__init__(resID, resPath, resType, resFormat)

    def load(self):
        # Preload all images ?
        pass

    def load_resource(self, filepath, boundary_values):
        with open(filepath, 'rb') as filepath:
            txt = filepath.read()
            return txt

class ImageResource(RawResource):
    """
    ImageResource: A collection of image files
    """
    def __init__(self, resID, resPath, resType, resFormat):
        self.keras_image = importlib.import_module('keras.preprocessing.image')
        self.pil_image = importlib.import_module('PIL.Image')
        super(ImageResource, self).__init__(resID, resPath, resType, resFormat)

    def load(self):
        # Preload all images ?
        pass

    def load_resource(self, filepath, boundary_values):
        im = self.keras_image.load_img(filepath, target_size=(224, 224))
        return dict(data=im.tobytes(), size=im.size, mode=im.mode)

    def unserialize_resource(self, value):
        return self.pil_image.frombytes(**value)

class AudioResource(RawResource):
    """
    AudioResource: A collection of audio files
    """
    def __init__(self, resID, resPath, resType, resFormat):
        self.librosa = importlib.import_module('librosa')
        super(AudioResource, self).__init__(resID, resPath, resType, resFormat)

    def load(self):
        # Preload all audio ?
        pass

    def load_resource(self, filepath, boundary_values):
        kwargs = {'sr':None}
        if len(boundary_values) == 2:
            start = float(boundary_values[0])
            end = float(boundary_values[1])
            if start > end:
                tmp = start
                start = end
                end = tmp
            kwargs['offset'] = start
            kwargs['duration'] = end-start
        try:
            return self.librosa.load(filepath, **kwargs)
        except:
            return None

class TimeSeriesResource(RawResource):
    """
    TimeSeriesResource: A collection of time series files.
    """
    def __init__(self, resID, resPath, resType, resFormat):
        super(TimeSeriesResource, self).__init__(resID, resPath, resType, resFormat)
        
        # FIXME: assume name of index is 'time'
        self.index_column = 'time'

    def load(self):
        # Preload all time series ?
        pass

    def load_resource(self, filepath, _):
        self.df = pd.read_csv(filepath, index_col=self.index_column)
        self.orig_df = copy.copy(self.df)        
        return self.df

class GraphResource(DataResource):
    graph = None

    def __init__(self, resID, resPath, resType, resFormat):
        self.networkx = importlib.import_module('networkx')
        super(GraphResource, self).__init__(resID, resPath, resType, resFormat)

    '''
    Initial Load.. Does nothing for now. Could preload
    '''
    def load(self):
        self.graph = self.networkx.read_gml(self.resPath)

    def join_with(self, resource, reference):
        assert(type(resource) is TableResource)
        colname = reference.from_column["colName"]
        toref = reference.to_reference
        '''
        resource.df[colname] = resource.df[colname].astype(object)
        for index, row in resource.df.iterrows():
            item = row[colname]
            values = None
            if toref == "node":
                # Get node specific stuff
                values = self.networkx.get_node_attributes(self.graph, item)
            elif toref == "edge":
                # Get edge specific stuff
                values = self.networkx.get_edge_attributes(self.graph, item)
            resource.df.at[index, colname] = values
        '''

class Reference(object):
    """
    Used to define a reference from one resource to another
    """
    from_resource = None
    from_column = None
    to_resource = None
    to_reference = None
    scanned = False

    def __init__(self, fromres, fromcol, tores, toref):
        self.from_resource = fromres
        self.from_column = fromcol
        self.to_resource = tores
        self.to_reference = toref

    def __repr__(self):
        torefstr = self.to_reference
        if type(self.to_resource) is TableResource:
            torefstr = self.to_reference["columnName"]
        return "{}.{}->{}.{}".format(
            self.from_resource.resID,
            self.from_column["colName"],
            self.to_resource.resID,
            torefstr)
