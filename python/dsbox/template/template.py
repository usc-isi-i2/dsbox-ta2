import copy
import typing

from itertools import product
from pprint import pprint


from d3m import container, exceptions, utils, index as d3m_index
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
        self.need_add_reference = False

        # Need to be set by subclass inheriting DSBoxTemplate
        self.template = {}

    def __str__(self):
        if hasattr(self, 'template') and 'name' in getattr(self, 'template'):
            return f"DSBoxTemplate:{self.template['name']}"
        else:
            return f"DSBoxTemplate:BLANK"

    def __repr__(self):
        return self.__str__()

    def validate(self):
        if not self.template:
            raise ValueError(f'Template not defined: {type(self).__name__}')
        # validate task
        # valiate subtask
        # valiate resource types
        # validate specialized problems
        # validate steps
        if 'steps' not in self.template:
            raise ValueError(f'Template {self.template["name"]} missing "steps" definitions')
        step_names: set = set()
        for i, each_step in enumerate(self.template['steps']):
            if 'name' not in each_step:
                raise ValueError(f'Template {self.template["name"]} step number {i} missing "name"')
            step_names.add(each_step['name'])
            if 'primitives' not in each_step:
                raise ValueError(f'Template {self.template["name"]} step {each_step["named"]}({i}) missing "primitives" list')
            for primitive in each_step['primitives']:
                self._validate_primitive(each_step, primitive, 0)

    def _validate_primitive(self, step, primitive, level):
        if level > 1:
            raise ValueError(f'Template {self.template["name"]} step {step["name"]}: Cannot have nest lists of primitives.')
        if isinstance(primitive, str):
            # name of primitive
            pass
        elif isinstance(primitive, dict):
            self._validate_primitive_desc(step, 0, primitive)
        elif isinstance(primitive, list):
            for i, p in enumerate(primitive):
                self._validate_primitive(step, p, level+1)

    def _validate_primitive_desc(self, step, i, primitive_desc: dict) -> None:
        if 'primitive' not in primitive_desc:
            raise ValueError(f'Template {self.template["name"]} step {step["name"]}({i}): missing "primitive" name.')
        if len(primitive_desc) > 1 and 'hyperparameters' not in primitive_desc:
            keys = set(primitive_desc.keys())
            keys.discard('primitive')
            raise ValueError(f'Template {self.template["name"]} step {step["name"]}({i}): contain extra key(s) {keys}')
        if 'hyperparameters' in primitive_desc:
            hyper: dict = primitive_desc['hyperparameters']
            for key, value in hyper.items():
                if not (isinstance(value, list) or isinstance(value, tuple)):
                    raise ValueError(f'Template {self.template["name"]} step {step["name"]}({i}) key ({key}) values must a list or a tuple')


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
            dstemp.to_pipeline('TESTING', configuration_point)
        """

        # add inputs to the configuration point
        ioconf = self.add_inputs_to_confPonit(configuration_point)

        # binding = configuration_point
        binding, sequence = self.add_intermediate_type_casting(ioconf)

        return self._to_pipeline(binding, sequence)

    def add_inputs_to_confPonit(self,
                                configuration_point: ConfigurationPoint) -> ConfigurationPoint:

        io_conf = copy.deepcopy(configuration_point)
        for step in self.template['steps']:
            io_conf[step['name']]['inputs'] = step['inputs']
        return io_conf

    def add_intermediate_type_casting(
            self, configuration_point: ConfigurationPoint) \
            -> typing.List:  # list of ConfigurationPoint and list
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

                # if list, assume it's okay
                if in_primitive_value is container.List and type(in_arg) is list:
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

            mystep = {
                "primitive": binding[name]["primitive"],
                "hyperparameters": binding[name]["hyperparameters"],
                "inputs": fill_in
            }

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

    def bind_primitive_IO(self, primitive: PrimitiveStep, templateIO):
        # print(templateIO)
        if len(templateIO) == 1:
            primitive.add_argument(
                name="inputs",
                argument_type=metadata_base.ArgumentType.CONTAINER,
                data_reference=templateIO[0])
        # if len(templateIO) > 1:
        else:
            arguments_train = primitive.primitive.metadata.query()['primitive_code']['instance_methods'][
                'set_training_data']['arguments']
            arguments_produce = primitive.primitive.metadata.query()['primitive_code']['instance_methods'][
                'produce']['arguments']

            arguments = []
            added = set()
            for t in arguments_train:
                arguments.append(t)
                added.add(t)
            for p in arguments_produce:
                if p not in added and p != 'timeout' and p != 'iterations':
                    arguments.append(p)
                    added.add(p)
            for index, argument in enumerate(arguments):
                primitive.add_argument(
                    name=argument,
                    argument_type=metadata_base.ArgumentType.CONTAINER,
                    data_reference=templateIO[index]
                )

    def _to_pipeline(self, binding, sequence) -> Pipeline:
        """
        Args:
            binding:

        Returns:

        """

        # define an empty pipeline with the general dataset input primitive
        # generate empty pipeline with i/o/s/u =[]
        pipeline = Pipeline(
            name=self.template['name'] + ":" + str(id(binding)),
            description=self.description_info,
            source={
                'name': 'ISI',
                'contact': 'mailto:kyao@isi.edu'
            })
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
            # no longer needed
            # if primitive_name == "d3m.primitives.data_augmentation.datamart_augmentation.DSBOX":
            #     hyper = binding[step]["hyperparameters"]
            #     primitive_step.add_argument("inputs1",metadata_base.ArgumentType.CONTAINER,"steps.0.produce")
            #     primitive_step.add_argument("inputs2",metadata_base.ArgumentType.CONTAINER, templateinput)
            #     for hyperName in hyper.keys():
            #         primitive_step.add_hyperparameter(
            #             # argument_type should be fixed type not the type of the data!!
            #             name=hyperName, argument_type=self.argmentsmapper["value"],
            #             data=hyper[hyperName])
            #     # pre v2019.1.21
            #     pipeline.add_step(primitive_step)
            #     primitive_step.add_output("produce")
            #     outputs[step] = f'steps.{primitive_step.index}.produce'
            #     continue

            # if primitive_name == "d3m.primitives.data_augmentation.datamart_query.DSBOX":
            #     primitive_step.add_argument("inputs",metadata_base.ArgumentType.CONTAINER, templateinput)
            #     hyper = binding[step]["hyperparameters"]
            #     for hyperName in hyper.keys():
            #         primitive_step.add_hyperparameter(
            #             # argument_type should be fixed type not the type of the data!!
            #             name=hyperName, argument_type=self.argmentsmapper["value"],
            #             data=hyper[hyperName])
            #     # pre v2019.1.21
            #     pipeline.add_step(primitive_step)
            #     primitive_step.add_output("produce")
            #     outputs[step] = f'steps.{primitive_step.index}.produce'
            #     continue


            if binding[step]["hyperparameters"] != {}:
                hyper = binding[step]["hyperparameters"]
                for hyperName in hyper.keys():
                    primitive_step.add_hyperparameter(
                        # argument_type should be fixed type not the type of the data!!
                        name=hyperName, argument_type=self.argmentsmapper["value"],
                        data=hyper[hyperName])

            # add reference to denormalized results if construct predictions steps added
            if primitive_name == 'd3m.primitives.data_transformation.construct_predictions.Common':
                primitive_step.add_argument("reference", metadata_base.ArgumentType.CONTAINER, "steps.0.produce")

            # first we need to extract the types of the primtive's input and
            # the generators's output type.
            # then we need to compare those and in case we have different
            # types, add the intermediate type caster in the pipeline
            # print(outputs)
            step_parameters = binding[step]["inputs"]
            step_arguments = []
            for parameter in step_parameters:
                if type(parameter) is list:
                    argument = [outputs[subparam] for subparam in parameter]
                else:
                    argument = outputs[parameter]
                step_arguments.append(argument)
            self.bind_primitive_IO(primitive_step, step_arguments)
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
            values: list = []

            # description: typing.Dict
            for description in each_step["primitives"]:
                value_step = []
                # primitive with no hyperparameters
                if isinstance(description, str):
                    value_step.append({
                        "primitive": description,
                        "hyperparameters": {}
                    })
                # one primitive with hyperparameters
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
            return value

        # 2019.3.25 update: Because the query of datamart is different,
        #                   We use dict as a hyperparameter, we have to do some special change here
        if description["primitive"] == "d3m.primitives.data_augmentation.datamart_query.DSBOX":
            value.append({
                    "primitive": description["primitive"],
                    "hyperparameters": description["hyperparameters"],
                })
            return value

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

        # iterate through all combinations of the hyperparameters and add
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



class DSBoxTemplate_Dragonfly(DSBoxTemplate):
    SEP = "__:__"

    def get_drgnfly_config(self) -> \
        typing.Dict[str, typing.Union[str, float, int]]:

        drgnfly_config = {"name": self.template['name'], "domain": {}}
        steps = self.template["steps"]

        # step_size: int = None
        for each_step in steps:
            step_name = each_step["name"]

            # prim_name = primitive_list['primitive']
            # prim_id = step_name + DRAGONFLY_CONFIG_SEP + prim_name

            primitive_list = each_step["primitives"]
            for step_prim_desc in primitive_list:
                # one primitive with no hyperparameter
                step_id = step_name+self.SEP
                if isinstance(step_prim_desc, str):
                    # entry = self._string_primitive(step_name, step_prim_desc)
                    drgnfly_config['domain'][step_id] = (
                        self.drgn_entry(n=step_id,t='discrete',
                                        it=[step_prim_desc])
                    )
                    # entry[1]
                # one primitive with hyperparamters
                elif isinstance(step_prim_desc, dict):
                    self._dict_primitive(drgnfly_config,
                                         step_prim_desc,
                                         step_id)

                # list of primitives
                elif isinstance(step_prim_desc, list):
                    added_prims = []
                    for one_prim in step_prim_desc:
                        if isinstance(one_prim, str):
                            added_prims.append(one_prim)
                        elif isinstance(one_prim, dict):
                            prim_id = self._dict_primitive(drgnfly_config,
                                                           one_prim,
                                                           step_id)
                            added_prims.append(prim_id)
                        else:
                            assert False
                    drgnfly_config['domain'][step_id] = added_prims
                else:
                    # other data format, not supported, raise error
                    print("Error: Wrong format of the description: "
                          "Unsupported data format found : ", type())
                    assert False

        return drgnfly_config

    def _dict_primitive(self, drgnfly_config, one_prim, step_id):
        prim_id = step_id + one_prim['primitive']

        assert isinstance(one_prim['hyperparameters'], dict)

        for h_name, h_vals in one_prim['hyperparameters'].items():
            hyper_id = (prim_id + self.SEP + h_name)
            assert isinstance(h_vals, list) and len(h_vals) > 0

            hyper_type = self.extract_hyper_type(h_vals)

            drgnfly_config['domain'][hyper_id] = (
                self.drgn_entry(n=step_id, t=hyper_type, it=h_vals)
            )
        return one_prim['primitive']

    def extract_hyper_type(self, h_vals: typing.List) -> str:
        if isinstance(h_vals[0], float):
            return 'float'
        elif isinstance(h_vals[0], int):
            return 'int'
        elif isinstance(h_vals[0], list):
            return 'discrete'
        return ('float' if isinstance(h_vals[0],
                                      float)
                else None)

    def drgnfly_config_to_confpoint(self, drgn_conf_p: typing.Dict[str, typing.Any])\
            -> ConfigurationPoint:

        assert 'domain' in drgn_conf_p and 'name' in drgn_conf_p

        assert isinstance(drgn_conf_p['domain'], dict)

        for e_name, e_val in drgn_conf_p['domain'].items():
            split_name = e_name.split(self.SEP)
            assert len(split_name) > 1

            # primitive options in each step
            if len(split_name) == 2:
                assert len(split_name[1]) == 0




    @staticmethod
    def drgn_entry (n: str, t: str, mi: float = None, mx: float = None,
                    it: list = None) -> typing.Dict[str, typing.Any]:
        assert isinstance(n, str)
        assert len(n) > 0

        assert t in ['int', 'float', 'discrete', 'discrete_numeric', 'boolean']

        if t in ['discrete_numeric', 'discrete']:
            assert it is not None
            assert isinstance(it, list)

            if t == 'discrete_numeric':
                assert all([isinstance(e, float) for e in it])

            assert mi is None and mx is None

            return {
                "name": n,
                "type": "discrete",
                "items": "-".join(it),
            }
        elif t in ['int', 'float']:
            assert it is None
            assert mi is not None and mx is not None
            return {
                "name": n,
                "type": t,
                "min": mi,
                "max": mx
            }
