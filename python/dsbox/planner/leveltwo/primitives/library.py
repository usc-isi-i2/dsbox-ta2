import json
import primitive
from dsbox.schema.profile_schema import DataProfileType as dpt

class PrimitiveLibrary(object):
    """
    Creates a Library of Primitives given the location of a library json
    """
    def __init__(self, location):
        self.primitives = []
        self.json = self.loadjson(location)
        for p in self.json:
            prim = primitive.Primitive(p['Name'], p['Class'])
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
