import copy
import numpy as np

from itertools import product
from pprint import pprint

from d3m import exceptions, utils, index as d3m_index
from d3m.metadata import base as metadata_base
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from .configuration_space import SimpleConfigurationSpace, ConfigurationPoint


class HyperparamDirective(utils.Enum):
    """
    Specify how to choose hyperparameters
    """
    DEFAULT = 1
    RANDOM = 2


class DSBoxTemplate():
    def __init__(self):
        self.primitive = d3m_index.search()
        self.argmentsmapper = {
            "container": metadata_base.ArgumentType.CONTAINER,
            "data": metadata_base.ArgumentType.DATA,
            "value": metadata_base.ArgumentType.VALUE,
            "primitive": metadata_base.ArgumentType.PRIMITIVE
        }
        self.stepcheck = None  # Generate a step check matrix

        self.step_number = {}
        self.addstep_mapper = {
            ("<class 'd3m.container.pandas.DataFrame'>",
             "<class 'd3m.container.numpy.ndarray'>"): "d3m.primitives.data.DataFrameToNDArray",
            # ("<class 'd3m.container.pandas.DataFrame'>", "<class 'd3m.container.numpy.ndarray'>"): "d3m.primitives.data_cleaning.imputer.SKlearn",
            ("<class 'd3m.container.numpy.ndarray'>",
             "<class 'd3m.container.pandas.DataFrame'>"): "d3m.primitives.data.NDArrayToDataFrame"
        }
        self.description_info = ""
        # Need to be set by subclass inheriting DSBoxTemplate
        # self.template = ""

    def __str__(self):
        if hasattr(self, 'template') and 'name' in getattr(self, 'template'):
            return f"DSBoxTemplate:{self.template['name']}"
        else:
            return f"DSBoxTemplate:BLANK"

    def __repr__(self):
        return self.__str__()

    def add_stepcheck(self):
        check = np.zeros(shape=(len(self.primitive), len(self.primitive))).astype(int)
        for i, v in enumerate(self.primitive.keys()):
            inputs = self.primitive[v].metadata.query()["primitive_code"]["class_type_arguments"][
                "Inputs"]
            for j, u in enumerate(self.primitive.keys()):
                outputs = self.primitive[u].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"]
                try:
                    inp = inputs.__args__
                    if outputs in inp:
                        check[i][j] = 1
                except Exception:
                    if inputs == outputs:
                        check[i][j] = 1
        self.stepcheck = check

    def to_pipeline(self, configuration_point: ConfigurationPoint) -> Pipeline:
        """
        converts the configuration point to the executable pipeline based on
        ta2 competitions format
        Args:
            configuration_point (ConfigurationPoint):

        Returns:
            The executable pipeline with full hyperparameter settings

        Examples:
            configuration_point =
            {
                "my_step1" : {
                    "primitive": "dsbox.a.b",
                    "hyperparameters": {
                        "x": 1
                    }
                },
                "my_step2" : {
                    "primitive": "sklearn.a.b",
                    "hyperparameters": {}
                }
            }
            dstemp = DSBoxTemplate(...)
            dstemp.to_pipeline(configuration_point)
        """
        # print("*" * 20)
        # print("[INFO] to_pipeline:")
        # pprint(configuration_point)
        # return self._to_pipeline(configuration_point)

        # add inputs to the configuration point
        ioconf = self.add_inputs_to_confPonit(configuration_point)

        # binding = configuration_point
        binding, sequence = self.add_intermediate_type_casting(ioconf)
        # print("[INFO] Binding:")
        # pprint(binding)
        return self._to_pipeline(binding, sequence)

    def add_inputs_to_confPonit(self,
                                configuration_point: ConfigurationPoint) -> ConfigurationPoint:

        io_conf = copy.deepcopy(configuration_point)
        for step in self.template['steps']:
            io_conf[step['name']]['inputs'] = step['inputs']
        return io_conf

    def add_intermediate_type_casting(
            self, configuration_point: ConfigurationPoint) \
            -> ConfigurationPoint:
        """
        This method parses the information in the template and adds the
        necessary type casting primitives in the pipeline. These type
        information is associated with each individual primitive present in
        the template and is governed by d3m's primitive rules.
        Args:
            configuration_point: Configuration

        Returns:
            binding: Configuration

        """
        # binding = ....
        binding = configuration_point
        checked_binding = {}
        sequence = []
        # for step in self.template["steps"]:
        for step_num, step in enumerate(self.template["steps"]):
            # First element in the inputs array is always the input of the
            # step in configuration point. In order to check the need for
            # adding intermediate step we first extract metadata information
            # of steps and by comparing the IO type information we decide on
            # whether intermediate type caster is necessary or not

            inputs = step["inputs"]
            fill_in = copy.deepcopy(inputs)
            name = step["name"]
            for in_arg in inputs:
                in_primitive_value = d3m_index.get_primitive(binding[name]["primitive"]).metadata.query()[
                    "primitive_code"]["class_type_arguments"]["Inputs"]

                if in_arg == "template_input":
                    continue

                # Check if the input name is valid and available in template
                if in_arg not in binding:
                    print("[ERROR] step {} input {} is not available!".format(step_num, in_arg))
                    print("binding: ")
                    pprint(binding)
                    return 1

                # get information of the producer of the input
                out_primitive_value = \
                    d3m_index.get_primitive(binding[in_arg]["primitive"]).metadata.query()[
                        "primitive_code"]["class_type_arguments"]["Outputs"]
                if not self.iocompare(in_primitive_value,
                                      out_primitive_value):
                    check_key = (str(out_primitive_value),
                                 str(in_primitive_value))
                    print("[INFO] Different types!")
                    try:
                        # inter_name = "{}_{}_{}".format(name,in_arg,solution)
                        solution = self.addstep_mapper[check_key]
                        inter_name = "{}_{}_{}".format(name, in_arg, solution)
                        intermediate_step = {
                            "primitive": solution,
                            "hyperparameters": {},
                            "inputs": [in_arg]
                        }
                        # binding[inter_name] = intermediate_step
                        # binding[name]['inputs'][0] = inter_name
                        # checked_binding[inter_name] = intermediate_step
                        pos = binding[name]["inputs"].index(in_arg)
                        # checked_binding[name]["inputs"][pos] = inter_name
                        checked_binding[inter_name] = intermediate_step
                        fill_in[pos] = in_arg
                        sequence.append(inter_name)
                        print("[INFO] ", solution, "added to step",
                              name)
                    except:
                        print("Warning!", name,
                              "'s primitive",
                              # Fixme:
                              # conf_step[-1]["primitive"],
                              "'s inputs does not match",
                              binding[in_arg][-1]["primitive"],
                              "and there is no converter found")

            # temporary fix for CMU clustering tempalte (with special input called "reference")
            need_add_reference = False
            for each_primitive in step['primitives']:
                if 'reference' in each_primitive:
                    need_add_reference = True

            if need_add_reference:
                mystep = {
                    "primitive": binding[name]["primitive"],
                    "hyperparameters": binding[name]["hyperparameters"],
                    "reference": step['primitives'][0]['reference'],
                    "inputs": fill_in
                }
            else:
                mystep = {
                    "primitive": binding[name]["primitive"],
                    "hyperparameters": binding[name]["hyperparameters"],
                    "inputs": fill_in
                }
            import pdb
            pdb.set_trace()
            if "runtime" in step:
                mystep["runtime"] = step["runtime"]

            sequence.append(name)
            checked_binding[name] = mystep

        return checked_binding, sequence

    def iocompare(self, i, o):
        try:
            i = i.__args__
            if (o in i) or (i in o):
                return True
        except Exception:
            if o == i:
                return True
        return False

    def bind_primitive_IO(self, primitive: PrimitiveStep, *templateIO):
        # print(templateIO)
        if len(templateIO) > 0:
            primitive.add_argument(
                name="inputs",
                argument_type=metadata_base.ArgumentType.CONTAINER,
                data_reference=templateIO[0])

        if len(templateIO) > 1:
            arguments = primitive.primitive.metadata.query()['primitive_code']['instance_methods'][
                'set_training_data']['arguments']
            if "outputs" in arguments:
                # Some primitives (e.g. GreedyImputer) require "outputs", while others do
                # not (e.g. MeanImputer)
                primitive.add_argument("outputs", metadata_base.ArgumentType.CONTAINER,
                                       templateIO[1])
        if len(templateIO) > 2:
            raise exceptions.InvalidArgumentValueError(
                "Should be less than 3 arguments!")

    def _to_pipeline(self, binding, sequence) -> Pipeline:
        """
        Args:
            binding:

        Returns:

        """

        # define an empty pipeline with the general dataset input primitive
        # generate empty pipeline with i/o/s/u =[]
        # pprint(binding)
        # print(sequence)
        # print("[INFO] list:",list(map(str, metadata_base.Context)))
        pipeline = Pipeline(name=self.template['name'] + ":" + str(id(binding)),
                            context=metadata_base.Context.PRETRAINING,
                            description=self.description_info)  # 'PRETRAINING'
        templateinput = pipeline.add_input("input dataset")

        # save temporary output for another step to take as input
        outputs = {}
        outputs["template_input"] = templateinput

        # iterate through steps in the given binding and add each step to the
        #  pipeline. The IO and hyperparameter are also handled here.
        for i, step in enumerate(sequence):
            self.step_number[step] = i
            # primitive_step = PrimitiveStep(self.primitive[binding[step][
            # "primitive"]].metadata.query())
            primitive_name = binding[step]["primitive"]
            if primitive_name in self.primitive:
                primitive_desc = dict(d3m_index.get_primitive(primitive_name).metadata.query())

                primitive_step = PrimitiveStep(primitive_desc)

                # D3M version v2019.1.21 removes primitive description. Need another way
                # to pass "runtime"
                if "runtime" in binding[step]:
                    # primitive_desc["runtime"] = binding[step]["runtime"]
                    primitive_step.__dict__['_dsbox_runtime'] = binding[step]["runtime"]
                    # print('==== ', primitive_step._dsbox_runtime)

            else:
                raise exceptions.InvalidArgumentValueError("Error, can't find the primitive : ",
                                                           primitive_name)

            if binding[step]["hyperparameters"] != {}:
                hyper = binding[step]["hyperparameters"]
                for hyperName in hyper:
                    primitive_step.add_hyperparameter(
                        # argument_type should be fixed type not the type of the data!!
                        name=hyperName, argument_type=self.argmentsmapper["value"],
                        data=hyper[hyperName])
            templateIO = binding[step]["inputs"]

            # first we need to extract the types of the primtive's input and
            # the generators's output type.
            # then we need to compare those and in case we have different
            # types, add the intermediate type caster in the pipeline
            # print(outputs)
            self.bind_primitive_IO(primitive_step,
                                   *map(lambda io: outputs[io], templateIO))
            pipeline.add_step(primitive_step)
            # pre v2019.1.21
            # outputs[step] = primitive_step.add_output("produce")
            primitive_step.add_output("produce")
            outputs[step] = f'steps.{primitive_step.index}.produce'
        # END FOR

        # Add final output as the prediction of target attribute
        general_output = outputs[self.template["steps"][-1]["name"]]
        # print(general_output)
        pipeline.add_output(general_output, "predictions of input dataset")

        return pipeline

    def generate_configuration_space(self) -> SimpleConfigurationSpace:
        steps = self.template["steps"]
        conf_space = {}
        for each_step in steps:
            name = each_step["name"]
            values = []

            # description: typing.Dict
            for description in each_step["primitives"]:
                value_step = []
                # primitive with no hyperparameters
                if isinstance(description, str):
                    value_step.append({
                        "primitive": description,
                        "hyperparameters": {}
                    })
                # one primitive with hyperparamters
                elif isinstance(description, dict):
                    value_step += self.description_to_configuration(description)
                # list of primitives
                elif isinstance(description, list):
                    for prim in description:
                        value_step += self.description_to_configuration(prim)
                else:
                    # other data format, not supported, raise error
                    print("Error: Wrong format of the description: "
                          "Unsupported data format found : ", type(description))

                values += value_step

            # END FOR
            if len(values) > 0:
                conf_space[name] = values
        # END FOR
        return SimpleConfigurationSpace(conf_space)

    def description_to_configuration(self, description):
        value = []
        # if the desciption is an dictionary:
        # it maybe a primitive with hyperparameters
        if "primitive" not in description:
            print("Error: Wrong format of the configuration space data: "
                  "No primitive name found!")
        else:
            if "hyperparameters" not in description:
                description["hyperparameters"] = {}

            # go through the hypers and if anyone has empty value just remove it
            hyperDict = dict(filter(lambda kv: len(kv[1]) > 0,
                                    description["hyperparameters"].items()))

            # go through the hyper values for single tuples and convert them
            # to a list with single tuple element
            hyperDict = dict(map(
                lambda kv:
                (kv[0], [kv[1]]) if isinstance(kv[1], tuple) else (kv[0], kv[1]),
                hyperDict.items()
            ))

            # iterate through all combinations of the hyperparamters and add
            # each as a separate configuration point to the space
            for hyper in _product_dict(hyperDict):
                value.append({
                    "primitive": description["primitive"],
                    "hyperparameters": hyper,
                })
        return value

    def get_target_step_number(self):
        # self.template[0].template['output']
        return self.step_number[self.template['output']]

    def get_output_step_number(self):
        return self.step_number[self.template['output']]


def _product_dict(dct):
    keys = dct.keys()
    vals = dct.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))
