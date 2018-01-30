import json
import os
import pprint
import typing

from typing import Dict, List, Union
from warnings import warn

import d3m 

from d3m_metadata.metadata import PrimitiveFamily

from .library import D3MPrimitiveLibrary
from dsbox.planner.common.primitive import Primitive

class Hierarchy(object):
    """Generic tree of nodes"""
    def __init__(self, name):
        # name of this hierarchy
        self.name = name
        self.root = HierarchyNode(self, 'root', None)
        self._changed = False
        self._level_count = []
        self.node_by_primitive : Dict[Primitive, HierarchyNode]= dict()

    def add_child(self, node, name, content=None):
        """Create and add child node"""
        assert node.hierarchy == self
        child = node._add_child(name, content)
        if content:
            for primitive in content:
                self.node_by_primitive[primitive] = node
        self._changed = True
        return child

    def add_path(self, names):
        """Create and add all nodes in path"""
        curr_node = self.root
        for name in names:
            if curr_node.has_child(name):
                curr_node = curr_node.get_child(name)
            else:
                curr_node = curr_node._add_child(name)
                self._changed = True
        return curr_node

    def add_primitive(self, node, primitive):
        self.node_by_primitive[primitive] = node
        node._add_primitive(primitive)

    def get_level_counts(self):
        """Computes the number of nodes at each level"""
        if not self._changed:
            return self._level_count
        self._level_count = self._compute_level_counts(self.root, 0, list())
        self._changed = False
        return self._level_count

    def _compute_level_counts(self, node, level, counts):
        """Computes the number of nodes at each level"""
        if len(counts) < level + 1:
            counts = counts + [0]
        counts[level] = counts[level] + 1
        for child in node._children:
            counts = self._compute_level_counts(child, level+1, counts)
        return counts

    def get_primitive_count(self):
        """Returns the number of primitives"""
        return self._get_primitive_count(self.root)

    def _get_primitive_count(self, curr_node):
        """Returns the number of primitives"""
        count = 0
        if curr_node._content:
            count = count + len(curr_node._content)
        for child in curr_node._children:
            count = count + self._get_primitive_count(child)
        return count

    def get_primitives(self, curr_node=None) -> List[List[Primitive]]:
        '''Returns list of list primitives of the subtree rooted at curr_node'''
        if curr_node is None:
            curr_node = self.root
        result = []
        if curr_node._content:
            result.append(curr_node._content)
        for child in curr_node._children:
            result = result + self.get_primitives(child)
        return result

    def get_primitives_as_list(self, curr_node=None) -> List[Primitive]:
        '''Returns list of primitives of the current node and of its immediate children'''
        return [p for plist in self.get_primitives(curr_node) for p in plist]

    def get_nodes_by_level(self, level):
        """Returns node at a specified level of the tree"""
        return self._get_nodes_by_level(self.root, 0, level)

    def _get_nodes_by_level(self, curr_node, curr_level, target_level):
        """Returns node at a specified level of the tree"""
        if curr_level >= target_level:
            return [curr_node]
        elif curr_level +1 == target_level:
            return curr_node._children
        else:
            result = []
            for node in curr_node._children:
                result = result + self._get_nodes_by_level(node, curr_level + 1, target_level)
            return result

    def has_primitive(self, primitive):
        return primitive in self.node_by_primitive
    
    def get_node_by_primitive(self, primitive):
        return self.node_by_primitive[primitive]

    def pprint(self):
        """Print hierarchy"""
        print('Hierarchy({}, level_counts={})'.format(self.name, self.get_level_counts()))
        self._print(self.root, [])

    def _print(self, curr_node, path, max_depth=2):
        new_path = path + [curr_node]
        if len(new_path) > max_depth:
            print(' '*4 + ':'.join([node.name for node in new_path[1:]]))
            for line in pprint.pformat(curr_node._content).splitlines():
                print(' '*8 + line)
        else:
            for child in curr_node._children:
                self._print(child, new_path, max_depth=max_depth)

    def __str__(self):
        return 'Hierarchy({}, num_primitives={}, level_node_counts={})'.format(
            self.name, self.get_primitive_count(), self.get_level_counts())

class HierarchyNode(object):
    """Node in the Hierarchy"""
    def __init__(self, hierarchy, name, parent, content=None):
        self.hierarchy = hierarchy
        self.name = name
        self._parent = parent
        self._children = []
        if content:
            self._content = content
        else:
            self._content = list()
    def getChildren(self):
        return self._children
    def get_content(self):
        return self._content
    def _add_child(self, name, content=None):
        """Add a child to the hierarchy node"""
        child = HierarchyNode(self.hierarchy, name, self, content)
        self._children.append(child)
        return child
    def has_child(self, name):
        """Return true if node has child with given name"""
        for node in self._children:
            if node.name == name:
                return True
        return False
    def get_child(self, name):
        """Return child by name"""
        for node in self._children:
            if node.name == name:
                return node
        raise Exception('Child not found: {}'.format(name))
    def get_siblings(self):
        if self._parent is None:
            result = []
        else:
            result = [x for x in self._parent._children if not x==self]
        return result
#    def add_primitive(self, primitive):
#        self.hierarchy.add_primitive(self, primitive)
    def _add_primitive(self, primitive):
        self._content.append(primitive)
    def __str__(self):
        return 'Node({},num_child={})'.format(self.name, len(self._children))
    def __repr__(self):
        return 'Node({},num_child={})'.format(self.name, len(self._children))
    
class D3MOntology(object):
    def __init__(self, library : D3MPrimitiveLibrary):
        self.library = library
        self.hierarchy = Hierarchy('D3M')
        for family in PrimitiveFamily:
            self.hierarchy.add_child(self.hierarchy.root, family.name)

    def load_curated_hierarchy(self, library_dir):
        filename = 'two_level_clustering-v{}.json'.format(d3m.__version__)
        
        self._load_curated_hierarchy(os.path.join(library_dir, filename))
        
    def _load_curated_hierarchy(self, hierarchy_file):
        with open(hierarchy_file) as pf:
            tree = json.load(pf)
        for key, value in tree['PrimitivesOntology'].items():
            if not self.hierarchy.root.has_child(key):
                warn('D3MOntology: New primitive family {}.'.format(key))
                self.hierarchy.add_child(self.hierarchy.root, key)
            self._add(self.hierarchy.root.get_child(key), value)
        for primitive in self.library.primitives:
            if not self.hierarchy.has_primitive(primitive):
                warn('D3MOnotology: primitive {} is NOT curated'.format(primitive.cls))
    
    def get_family(self, family) -> HierarchyNode:
        '''Returns node corresponding to the primitive family'''
        #  family : typing.Union(PrimitiveFamily, str)
        if not isinstance(family, str):
            family = family.name
        return self.hierarchy.root.get_child(family)
    
              
    def _add(self, node : HierarchyNode, spec : dict):
        for key, value in spec.items():
            child = self.hierarchy.add_child(node, key)
            if type(value) is dict:
                # add subtree
                self._add(child, value)
            else:
                # add list of primitives
                package_paths = set()
                for primtive_spec in value:
                    package_paths.add(primtive_spec.split('/')[2])
                for package in package_paths:
                    if self.library.has_primitive_by_package(package):
                        self.hierarchy.add_primitive(child, self.library.get_primitive_by_package(package))
                    else:
                        warn('D3MOnotology: primitive {} not found in library.'.format(package))    

    
