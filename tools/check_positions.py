"""Visualize sound heat map position parameters defined in Dockerfile"""

import ast
import atexit
import os
import sys

import dockerfile
import numpy as np
from bokeh.io import show
from bokeh.models import GMapOptions
from bokeh.plotting import gmap

# load docker file ENVs for MQTT broker parameters
df = dockerfile.parse_file("../Dockerfile")
for cmd in df:
    if cmd.cmd == "env":
        key = cmd.value[0]
        value = cmd.value[1]
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        os.environ[key] = value

sources_latlon = np.array(ast.literal_eval(os.environ["SOURCES"])).T
walls_latlon = np.array(ast.literal_eval(os.environ["WALLS"]))
area_latlon = np.array(ast.literal_eval(os.environ["AREA_POLYGON"])).T

map_options = GMapOptions(
    lat=sources_latlon[0, 0],
    lng=sources_latlon[1, 0],
    zoom=15,
    map_type="hybrid",
    tilt=0,
)

gmap_api_key = None
if gmap_api_key is None:
    raise NotImplementedError("Add gmaps api key in source.")

p = gmap(
    gmap_api_key, map_options, title="Tivoli"
)

p.circle(
    sources_latlon[1], sources_latlon[0], size=15, fill_color="blue", fill_alpha=0.8
)
p.multi_line(list(walls_latlon[:, :, 1]), list(walls_latlon[:, :, 0]), line_width=5)
p.patch(area_latlon[1], area_latlon[0], fill_alpha=0.5)

show(p)

def cleanup():
    os.chdir(sys.path[0])
    os.remove("gmaps.html")


atexit.register(cleanup)
