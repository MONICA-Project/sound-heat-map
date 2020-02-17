FROM python:3.7-stretch

##### CONFIGURATION #####

# Identifier for this sound heat map module
ENV SHM_ID "SHM_1"

# Configure access to GOST instance
ENV GOST_URL "http://monica-cloud.eu:5050/gost/v1.0"
ENV GOST_USER "ADD YOUR GOST USER HERE"
ENV GOST_PASS "ADD YOUR GOST PASSWORD HERE"

ENV MQTT_BROKER_PREFIX "GOST"
ENV CATALOG_URL "ADD YOUR SERVICE CATALOGE URL HERE"

### CONFIGURE SOUND HEAT MAP INPUT

# Optional filter for OGC SensorThingStore
ENV FILTER_MICS ""
# ENV FILTER_MICS "endswith(name, 'MONICA0010_100090_Id1')"
# ENV FILTER_MICS "startswith(name, 'SLM-GW')"

### CONFIGURE SOUND HEAT MAP OUTPUT

# Frequencies for which sound heat maps are created
ENV FREQUENCIES "[13, 16, 20, 25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]"

# Temperature used in sound propagation model
ENV TEMPERATURE "20.0"

# Location of sound sources in Lat/Lon coordinates
ENV SOURCES "[[45.0619, 7.6801], [45.0619, 7.68], [45.0619, 7.68005]]"

# Direction of sound sources
# Given in radiance. 0 points North. If empty string (""), monopole soures are used. Otherwise,
# dipole sources are used.
ENV SOURCES_DIRECTION "[-1.51417453, -1.51417453, -1.51417453]"

# Position of reflecting walls in Lat/Lon coordinates.
# Given as list of list of two points (starting point, end point).
ENV WALLS "[[[45.055, 7.675], [45.0619, 7.675]]]"

# Sound heat map area
# Sound heat map is predicted inside this polygon. Only datastreams of microphones that are inside this polygon are used.
ENV AREA_POLYGON "[[45.056, 7.676], [45.06, 7.676], [45.061, 7.685], [45.055, 7.685]]"

# Distance between sample points of the sound heat map
ENV CELLSIZE 0.0005

# Create one sound heat map stream for each of the weightings in WEIGHTINGS.
# Supportinged weightings are 'A', 'C', 'Z', 'Afullband', 'Cfullband', 'Zfullband'.
# In the 'Xfullband' form, the weighted sound pressure levels are summed to give a single number instead of
# one value for each value in FREQUENCIES
ENV WEIGHTINGS "['Z', 'Afullband', 'Cfullband']"

# Time in seconds between queries to GOST instance for new datastreams
ENV UPDATE_DATASTREAMS_PERIODE 60

# Time between computation and publishing of new sound heat maps
ENV NEW_MAP_PERIODE 10

# Sound heat maps are computed from the buffered data of the datastreams, between the times
# t1 = NOW - MAP_LATENCY
# and
# t2 = NOW - MAP_LATENCY - MAP_AVERAGING_TIME
ENV MAP_LATENCY 10
ENV MAP_AVERAGING_TIME 10


##### BUILDING CONTAINER #####

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app
WORKDIR /app

# CMD ["python", "shm.py", "--logging-level", "debug"]
CMD ["python", "shm.py"]
