import json
import primitive
from ..schema.profile import Profile

class PrimitiveLibrary(object):
    """
    Creates a Library of Primitives given the location of a library json
    """
    def __init__(self, location):
        self.primitives = []
        self.json = self.loadjson(location)
        for p in self.json:
            prim = primitive.Primitive(p['name'], p['class'])
            for precstr in p.get('requirements', []):
                prec = self.parseProfile(precstr)
                if prec:
                    prim.addPrecondition(prec)
            for effectstr in p.get('effects', []):
                effect = self.parseProfile(effectstr)
                if effect:
                    prim.addEffect(effect)                
            prim.types = p.get('type', [])
            prim.task = p.get('task', None)
            prim.column_primitive = p.get('column', False)
            self.primitives.append(prim)
    
    def parseProfile(self, profile): 
        neg = False
        if profile.startswith('!'):
            neg = True
            profile = profile[1:]
        if profile == "NUMERICAL":
            return Profile.NO_NUMERICAL if neg else Profile.NUMERICAL
        elif profile == "MISSING_VALUES":
            return Profile.NO_MISSING_VALUES if neg else Profile.MISSING_VALUES
        elif profile == "NON_NEGATIVE":
            return Profile.NO_NON_NEGATIVE if neg else Profile.NON_NEGATIVE
        return None 
            
    def loadjson(self, jsonfile):
        with open(jsonfile) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d
    
    def getPrimitivesByEffect(self, effect):
        plist = [];
        for primitive in self.primitives:
            for peffect in primitive.effects:
                if peffect == effect:
                    plist.append(primitive)
                    break
        return plist
