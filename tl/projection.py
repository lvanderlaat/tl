""" Projection

Handles with the coordinates tranformations.

"""

from pyproj import Proj, transform


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def geographic_to_cartesian(longitude, latitude, epsg):
    proj1 = Proj(init='epsg:4326')
    proj2 = Proj(init=f'epsg:{epsg}')
    return transform(proj1, proj2, longitude, latitude)


def cartesian_to_geographic(x, y, epsg):
    proj1 = Proj(init=f'epsg:{epsg}')
    proj2 = Proj(init='epsg:4326')
    return transform(proj1, proj2, x, y)
