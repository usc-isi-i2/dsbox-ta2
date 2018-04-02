import json
import os

from datetime import date
from typing import List, Dict

from d3m_metadata.metadata import PrimitiveMetadata, PrimitiveFamily, PrimitiveAlgorithmType
from d3m import index

from dsbox.planner.common.primitive import Primitive
from dsbox.schema.profile_schema import DataProfileType as dpt
from collections import defaultdict

class D3MPrimitiveLibrary(object):
    '''Creates a primitive library based on primitives_repo or d3m.index'''
    def __init__(self):
        self.api_version = ''
        # List of all primitives, except those in black list
        self.primitives : List[Primitive] = []

        # List of black listed primitives, e.g. pickling problems
        self.black_list_package : List[str] = []
        
        self.primitive_by_package : Dict[str, Primitive] = {}
        self.primitives_by_family : Dict[PrimitiveFamily, List[Primitive]] = defaultdict(list)
        self.primitives_by_type : Dict[PrimitiveAlgorithmType, List[Primitive]] = defaultdict(list)

    def has_api_version(self, primitives_repo_dir, api_version):
        return api_version in os.listdir(primitives_repo_dir)

    def load_from_directory(self, primitives_repo_dir, api_version=''):
        '''Load primitive description from filesystem.
         E.g. from repo https://gitlab.datadrivendiscovery.org/jpl/primitives_repo'''
        # Use fully for debugging

        listing = os.listdir(primitives_repo_dir)
        if api_version:
            if not api_version in listing:
                raise ValueError('API version {} not found')
        else:
            date_str = [x[1:] for x in listing if x.startswith('v')]
            if not date_str:
                raise ValueError('No API version found under {}'.format(primitives_repo_dir))
            dates = [date(*(map(int, x.split('.')))) for x in date_str]
            vdate = sorted(dates)[-1]
            api_version = 'v{}.{}.{}'.format(vdate.year, vdate.month, vdate.day)
        self.api_version = api_version

        api_dir = os.path.join(primitives_repo_dir, self.api_version)
        for team in os.listdir(api_dir):
            team_dir = os.path.join(api_dir, team)
            for module in os.listdir(team_dir):
                module_dir = os.path.join(team_dir, module)
                version = self._get_latest_version(os.listdir(module_dir))
                primitive_file = os.path.join(module_dir, version, 'primitive.json')
                with open(primitive_file) as fp:
                    d3m_metadata = PrimitiveMetadata(json.load(fp))
                    primitive = self._create_primitive_desc(d3m_metadata)
                    if primitive.cls in self.black_list_package:
                        print('Black listing primitive: {}'.format(primitive.name))
                    else:
                        self.primitives.append(primitive)
                    print('Primitive.cls: ', primitive.cls)
        self._setup()

    def load_from_d3m_index(self):
        '''Load primitive description from installed python packages'''
                
        for primitive_path, primitive_type in index.search().items():
            primitive = self._create_primitive_desc(primitive_type.metadata)
            if primitive.cls in self.black_list_package:
                print('Black listing primitive: {}'.format(primitive.name))
            else:
                self.primitives.append(primitive)

        self._setup()


    def get_primitives_by_family(self, family : PrimitiveFamily) -> List[Primitive]:
        return self.primitives_by_family[family]

    def has_primitive_by_package(self, path):
        return path in self.primitive_by_package

    def get_primitive_by_package(self, path):
        return self.primitive_by_package[path]

    def augment_with_primitive_profiler(self, profiler_json_file):
        '''Augment primitive with its requirements using Daniel's primitive profiler output'''
        with open(profiler_json_file) as fp:
            primitive_profiles = json.load(fp)

        for package, profile in primitive_profiles.items():
            if not self.has_primitive_by_package(package):
                print('Cannot find class: {}'.format(package))
                continue
            primitive = self.get_primitive_by_package(package)
            if 'Requirements' in profile:
                # Note: Cannot use {PrimitivePrecodition[x] : True for x in ...}, because extra "POSTIVE_VALUES"
                primitive.addPrecondition({x : True
                                           for x in profile['Requirements']})
            if 'Error' in profile:
                primitive.addErrorCondition({x:True for x in profile['Error']})

    def add_custom_primitive(self, class_str):
        mod, cls = class_str.rsplit('.', 1)
        try:
            import importlib
            module = importlib.import_module(mod)
            primitive_type = getattr(module, cls)
            primitive = self._create_primitive_desc(primitive_type.metadata)

            # Modify to actual python path
            primitive.cls = class_str
            self.primitives.append(primitive)
            self.primitive_by_package[class_str] = primitive
            return primitive
        except Exception as e:
            print('Failed to add primitive: {}'.format(e))
            return None

    def _get_latest_version(self, versions : List[str]):
        version_tuples = [v.split('.') if not v.startswith('v') else v[1:].split('.') for v in versions]
        version_tuples = list(map(lambda x : list(map(int, x)), version_tuples))
        latest_tuple = sorted(version_tuples)[-1]
        index = version_tuples.index(latest_tuple)
        return versions[index]

    def _create_primitive_desc(self, d3m : PrimitiveMetadata):
        primitive = Primitive(d3m.query()['id'], d3m.query()['name'], d3m.query()['python_path'])
        primitive.d3m_metadata = d3m
        return primitive

    def load_black_list(self, jsonfile):
        """Black list primitives that do not work properly"""
        with open(jsonfile) as json_data:
            black_list = json.load(json_data)
            names = []
            for pdict in black_list:
                # pid = pdict['Id']
                name = pdict["Name"]
                cls = pdict["Class"]
                self.black_list_package.append(cls)
                names.append(name)
        print('Primitives to black list: {}'.format(names))

    def is_black_listed(self, cls):
        return cls in self.black_list_package

    def _setup(self):
        for p in self.primitives:
            self.primitive_by_package[p.cls] = p
            self.primitives_by_family[p.getFamily()].append(p)
            types = p.getAlgorithmTypes()
            if isinstance(types[0], str):
                self.primitives_by_type[p.getAlgorithmTypes()[0]].append(p)
            else:
                self.primitives_by_type[p.getAlgorithmTypes()[0].value].append(p)

class PrimitiveLibrary(object):
    """
    Creates a Library of Primitives given the location of a library json
    """
    def __init__(self, location):
        self.primitives = []
        self.json = self.loadjson(location)
        for p in self.json:
            prim = Primitive(p['Id'], p['Name'], p['Class'])
            for precstr in p.get('Requirements', []):
                prec = self.parseProfile(precstr)
                if prec:
                    prim.addPrecondition(prec)
            for effectstr in p.get('Effects', []):
                effect = self.parseProfile(effectstr)
                if effect:
                    prim.addEffect(effect)
            prim.type = p.get('LearningType', None)
            prim.task = p.get('Task', None)
            prim.column_primitive = p.get('RequiresColumnData', False)
            prim.is_persistent = (prim.task == "Modeling") or (not p.get('NotPersistent', False))
            prim.unified_interface = p.get('UnifiedInterface', False)
            prim.init_args = p.get('InitArguments', [])
            prim.init_kwargs = p.get('InitKeywordArguments', {})
            self.primitives.append(prim)

    def parseProfile(self, profile):
        value = True
        if profile.startswith('!'):
            value = False
            profile = profile[1:]
        if hasattr(dpt, profile):
            return {getattr(dpt, profile): value}
        return None

    def loadjson(self, jsonfile):
        with open(jsonfile) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d

    def getPrimitivesByEffect(self, effect, value):
        plist = [];
        for primitive in self.primitives:
            if (primitive.preconditions.get(effect, False) != value and
                    primitive.effects.get(effect, False) == value):
                plist.append(primitive)
        return plist
