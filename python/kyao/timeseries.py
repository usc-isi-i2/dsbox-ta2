import os.path

from d3m.container.dataset import get_dataset

from d3m.primitives.data_transformation.denormalize import Common as Denormalize
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFramePrimitive
from d3m.primitives.data_transformation.extract_columns_by_semantic_types import DataFrameCommon as ExtracColumns
from d3m.primitives.data_transformation.column_parser import DataFrameCommon as ColumnParser
from d3m.primitives.time_series_forecasting.arima import DSBOX as Arima

dataset_dir = '/lfs1/ktyao/DSBox/data/datasets/seed_datasets_current/'
stock_train = get_dataset(os.path.join(dataset_dir, 'LL1_736_stock_market/TRAIN/dataset_TRAIN/datasetDoc.json'))
stock_test = get_dataset(os.path.join(dataset_dir, 'LL1_736_stock_market/TEST/dataset_TEST/datasetDoc.json'))

sunspots_train = get_dataset(os.path.join(dataset_dir, '56_sunspots/TRAIN/dataset_TRAIN/datasetDoc.json'))
sunspots_test = get_dataset(os.path.join(dataset_dir, '56_sunspots/TEST/dataset_TEST/datasetDoc.json'))

sunspots_monthly_train = get_dataset(os.path.join(dataset_dir, '56_sunspots_monthly/TRAIN/dataset_TRAIN/datasetDoc.json'))
sunspots_monthly_test = get_dataset(os.path.join(dataset_dir, '56_sunspots_monthly/TEST/dataset_TEST/datasetDoc.json'))

spawn_train = get_dataset(os.path.join(dataset_dir, 'LL1_736_population_spawn/TRAIN/dataset_TRAIN/datasetDoc.json'))
spawn_test = get_dataset(os.path.join(dataset_dir, 'LL1_736_population_spawn/TEST/dataset_TEST/datasetDoc.json'))

def steps(inputs):
    denormalize = Denormalize(hyperparams=Denormalize.metadata.get_hyperparams().defaults())
    dataset = denormalize.produce(inputs=inputs).value

    to_dataframe = DatasetToDataFramePrimitive(hyperparams=DatasetToDataFramePrimitive.metadata.get_hyperparams().defaults())
    raw_data = to_dataframe.produce(inputs=dataset).value

    numeric_columns = raw_data.metadata.list_columns_with_semantic_types(["http://schema.org/Integer", "http://schema.org/Float"])
    cp_hyperparams = ColumnParser.metadata.get_hyperparams().defaults()
    parse_type = ('http://schema.org/Boolean', 'http://schema.org/Integer', 'http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/FloatVector')
    cp_hyperparams = cp_hyperparams.replace({'parse_semantic_types': parse_type})
    parser = ColumnParser(hyperparams=cp_hyperparams)
    data = parser.produce(inputs=raw_data).value

    target_hyperparams=ExtracColumns.metadata.get_hyperparams().defaults()
    target_hyperparams = target_hyperparams.replace(
        {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',
                            'https://metadata.datadrivendiscovery.org/types/SuggestedTarget')})
    extract_target = ExtracColumns(hyperparams=target_hyperparams)
    target = extract_target.produce(inputs=data).value

    attr_hyperparams=ExtracColumns.metadata.get_hyperparams().defaults()
    attr_hyperparams = attr_hyperparams.replace(
        {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                            'https://metadata.datadrivendiscovery.org/types/Attribute')})
    extract_attribute = ExtracColumns(hyperparams=attr_hyperparams)
    attributes = extract_attribute.produce(inputs=data).value

    return target, attributes


# spawn
train = spawn_train
test = spawn_test
train_target, train_attributes = steps(train)
arima = Arima(hyperparams=Arima.metadata.get_hyperparams().defaults())
arima.set_training_data(inputs=train_attributes, outputs=train_target)
fit_result = arima.fit()

test_target, test_attributes = steps(test)
produce_results = arima.produce(inputs=test_attributes)
produce_results.value.to_csv('spawn_prediction.csv')

# sunspots_monthly
train = sunspots_monthly_train
test = sunspots_monthly_test
train_target, train_attributes = steps(train)
hyperparams = Arima.metadata.get_hyperparams().defaults()
hyperparams = hyperparams.replace({'take_log': False})
arima = Arima(hyperparams=hyperparams)
arima.set_training_data(inputs=train_attributes, outputs=train_target)
fit_result = arima.fit()

test_target, test_attributes = steps(test)
produce_results = arima.produce(inputs=test_attributes)
produce_results.value.to_csv('sunspots_monthly_prediction.csv')


# sunspots
train = sunspots_train
test = sunspots_test
train_target, train_attributes = steps(train)
arima = Arima(hyperparams=Arima.metadata.get_hyperparams().defaults())
arima.set_training_data(inputs=train_attributes, outputs=train_target)
fit_result = arima.fit()

test_target, test_attributes = steps(test)
produce_results = arima.produce(inputs=test_attributes)
produce_results.value.to_csv('sunspots_prediction.csv')


# stock
train = stock_train
test = stock_test
train_target, train_attributes = steps(train)
arima = Arima(hyperparams=Arima.metadata.get_hyperparams().defaults())
arima.set_training_data(inputs=train_attributes, outputs=train_target)
fit_result = arima.fit()

test_target, test_attributes = steps(test)
produce_results = arima.produce(inputs=test_attributes)
produce_results.value.to_csv('stock_prediction.csv')
