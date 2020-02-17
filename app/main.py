"""Sound Heat Map generation and distribution."""

import ast
import asyncio
import json
import logging
import optparse
import os
import re
import sys
import traceback
from datetime import datetime, timedelta

import coloredlogs
import dateutil.parser as dp
import numpy as np
import pandas as pd
import requests
from beeprint import pp
from hbmqtt.client import ClientException, MQTTClient
from hbmqtt.mqtt.constants import QOS_1
from pyost import SensorThingStore
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from model import calculate_shm_dir, grid_area
from weighting import weightfunc

logging.getLogger("urllib3").setLevel(logging.WARNING)  # not from these
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("hbmqtt").setLevel(logging.WARNING)
logging.getLogger("transitions").setLevel(logging.WARNING)

FREQUENCIES = np.array(ast.literal_eval(os.environ["FREQUENCIES"]))
MQTT_BROKER_PREFIX = os.environ["MQTT_BROKER_PREFIX"]
GOST_URL = os.environ["GOST_URL"]
UPDATE_DATASTREAMS_PERIODE = float(os.environ["UPDATE_DATASTREAMS_PERIODE"])
NEW_MAP_PERIODE = float(os.environ["NEW_MAP_PERIODE"])
MAP_LATENCY = float(os.environ["MAP_LATENCY"])
MAP_AVERAGING_TIME = float(os.environ["MAP_AVERAGING_TIME"])
FILTER_MICS = os.environ["FILTER_MICS"]
WEIGHTINGS = ast.literal_eval(os.environ["WEIGHTINGS"])

try:
    SHM_ID = os.environ["SHM_ID"]
except KeyError:
    # No instance set
    SHM_ID = None

try:
    AUTH = (os.environ["GOST_USER"], os.environ["GOST_PASS"])
except KeyError:
    AUTH = None


def CPBLZeq_Observation_to_dataframe(message):
    """Convert MQTT CPBLZeq observation message to a time indexable Dataframe."""
    # multiple observations can be bundled in one message
    observations = json.loads(message.data)["result"]["response"]["value"]

    all_values = []
    all_times = []
    for obs in observations:
        values = np.array(obs["values"])
        nmeas, nfreq = values.shape
        assert nfreq == 33

        start = dp.parse(obs["startTime"])
        end = dp.parse(obs["endTime"])

        times = start + (end - start) / nmeas * np.arange(nmeas)

        all_values.append(values)
        all_times.append(times)

    all_values = np.concatenate(all_values, axis=0)
    all_times = np.concatenate(all_times)

    return pd.DataFrame(all_values, index=all_times, columns=FREQUENCIES)


class Datastream:
    """Object representation of a Datastream."""

    def __init__(self, json):
        self.id = json["@iot.id"]
        self.location = json["observedArea"]["coordinates"]
        # coordinates must be given in (x=lon, y=lat) format, but we use lat/lon intern.
        self.location = self.location[::-1]
        self.description = json["description"]
        self.name = json["name"]
        self.mqtt_topic = MQTT_BROKER_PREFIX + f"/Datastreams({self.id})/Observations"
        self.recent_data = None

        if self.name.endswith("MONICA0010_100090_Id1"):
            logging.warning(f"Using hardcoded positions for {self.name}.")
            self.location = [45.797_271, 4.952_230]

    def add(self, data):
        self.recent_data = pd.concat((self.recent_data, data))

    def average_and_drop_old_data(self, starttime, endtime):
        # average
        if self.recent_data is None or self.recent_data.empty:
            return None

        avg = self.recent_data[starttime:endtime].mean(axis=0)

        logging.debug(
            "datastream.recent_data before drop:" + pp(self.recent_data, output=False)
        )

        # drop data that will never be used again
        self.recent_data = self.recent_data[starttime:]
        logging.debug(
            "datastream.recent_data after drop:" + pp(self.recent_data, output=False)
        )

        # return as ndarray
        return avg.values


class DatastreamManager:
    """Stores and updates active SLM Datastreams."""

    def __init__(self, store_url, mqtt_client, filter_mics, auth=None):
        self.st_store = SensorThingStore(store_url, auth=auth)
        self.datastreams = {}
        self.mqtt_client = mqtt_client
        self.filter_mics = filter_mics

    async def update_datastreams(self):
        """Query for available SLM datastreams and subscribe."""
        logging.info("Updating list of datastreams.")

        # query for ObservedProperty CPBLZeq
        observed_property = self.st_store.get_by_query(
            "ObservedProperties", filter="equals(name, 'CPBLZeq')"
        )[0]

        # query for all Datastreams that observe above property
        all_datastreams = self.st_store.get_by_parent(
            kind="Datastreams",
            parent="ObservedProperties",
            id=observed_property["@iot.id"],
            # filter="startswith(name, 'SLM-GW')",
            # filter="startswith(name, 'SDN-PLATFORM')",
            filter=self.filter_mics,
        )
        if not all_datastreams:
            logging.warning(
                f"Could not find any datastreams with filter {self.filter_mics}."
            )
            return

        # populate dictionary with new datastreams
        polygon = Polygon(np.array(ast.literal_eval(os.environ["AREA_POLYGON"])))
        for jso in all_datastreams:
            stream = Datastream(jso)
            if stream.id not in self.datastreams:

                # only include streams with location in area_polygon
                if polygon.contains(Point(stream.location)):
                    self.datastreams[stream.id] = stream
                else:
                    logging.info(
                        f"Excluding datastream {stream.id} with name {stream.name} at {stream.location}: not in polygon."
                    )

        if not self.datastreams:
            logging.info("No datastreams matching criteria found.")
            return

        # subscribe to all datastreams
        topics = [(ds.mqtt_topic, QOS_1) for (_, ds) in self.datastreams.items()]

        suback_codes = await self.mqtt_client.subscribe(topics)
        for code, topic in zip(suback_codes, topics):
            if code not in [0, 1, 2]:
                logging.error(
                    f"""SUBACK return code {code}. Could not subscribe to
                    topic: {topic}"""
                )
            else:
                logging.info(f"Subscribed to topic: {topic}")

    async def start(self, rate=UPDATE_DATASTREAMS_PERIODE):
        """Continually update datastream subscriptions."""
        while True:
            await self.update_datastreams()
            logging.debug("Datastreams: {}".format(pp(self.datastreams, output=False)))
            await asyncio.sleep(rate)


class DataCollector:
    """Collects MQTT messages and their data."""

    def __init__(self, mqtt_client, datastreams):
        self.mqtt_client = mqtt_client
        self.datastreams = datastreams

    def process(self, message):
        """Add contents of message to datastore."""
        # keep track of datastreams using their iot.id
        datastream_id = int(re.search(r"\d+", message.topic).group())
        logging.debug(
            f"Processing message from Datastream({datastream_id})"
            + f" from topic {message.topic}"
        )

        data = CPBLZeq_Observation_to_dataframe(message)

        if datastream_id in self.datastreams:
            self.datastreams[datastream_id].add(data)
        else:
            logging.error(
                f"Received message from Datastream({datastream_id}), altough not subscribed ..."
            )

        logging.debug(
            "Added data to datastreams:"
            + pp(self.datastreams[datastream_id], output=False)
        )

    async def start(self):
        """Listen to SLM observations and process incoming messages."""

        timeout = 60

        while True:

            try:
                message = await self.mqtt_client.deliver_message(timeout=timeout)
                logging.info(f"Received message on {message.topic}")
                logging.debug(f"Payload: {pp(json.loads(message.data), output=False)}")

                self.process(message)

            except ClientException as ce:
                logging.error("Client exception: %s" % ce)
            except asyncio.TimeoutError:
                logging.error(
                    f"Timeout error: did not receive message since {timeout}s"
                )


class MapMaker:
    def __init__(self, datastreams, mqtt_client, mqtt_shm_topics):
        self.datastreams = datastreams
        self.mqtt_client = mqtt_client
        self.mqtt_shm_topics = mqtt_shm_topics

    async def make_map(self):
        now = datetime.utcnow()
        starttime = now - timedelta(
            seconds=MAP_LATENCY + NEW_MAP_PERIODE + MAP_AVERAGING_TIME
        )
        endtime = now - timedelta(seconds=MAP_LATENCY)

        logging.debug(f"starttime: {starttime}, endtime: {endtime}")

        spl_values = []
        slm_latlon = []
        for (_, ds) in self.datastreams.items():
            avg = ds.average_and_drop_old_data(starttime, endtime)
            if avg is not None:
                # there was data to average in that microphone
                spl_values.append(avg)
                slm_latlon.append(ds.location)

        if not spl_values:
            logging.info("Couldn't create map: no data.")
            return None

        logging.info("Making a map.")

        slm_latlon = np.array(slm_latlon).T  # to shape 2 x Nslm
        spl_values = np.array(spl_values).T  # to shape Nf x Nslm

        logging.debug(
            pp({"Locations": slm_latlon, "averages": spl_values}, output=False)
        )

        temperature = float(os.environ["TEMPERATURE"])
        s_latlon = np.array(
            ast.literal_eval(os.environ["SOURCES"])
        ).T  # to shape 2 x Ns
        walls = os.environ["WALLS"]
        if walls:
            wall_latlon = np.array(
                ast.literal_eval(os.environ["WALLS"])
            ).T  # to shape 2 x 2 x Nw
        else:
            wall_latlon = None
        polygon = np.array(ast.literal_eval(os.environ["AREA_POLYGON"])).T
        cellsize = float(os.environ["CELLSIZE"])
        source_direction = np.array(ast.literal_eval(os.environ["SOURCES_DIRECTION"]))

        assert s_latlon.ndim == 2
        assert s_latlon.shape[0] == 2
        if walls:
            assert wall_latlon.ndim == 3
            assert wall_latlon.shape[0] == 2 and wall_latlon.shape[1] == 2
        assert polygon.ndim == 2
        assert polygon.shape[0] == 2
        assert isinstance(cellsize, float)
        assert FREQUENCIES.size == spl_values.shape[0]
        assert isinstance(temperature, float) or isinstance(temperature, int)

        # make a grid of points that includes the polygon
        r_latlon_all, mask_inside_polygon, grid_shape = grid_area(
            polygon, dx=cellsize, dy=cellsize
        )
        r_latlon_inside_polygon = r_latlon_all[:, mask_inside_polygon]

        # inside the polygon, one has the computed values
        try:
            shm = calculate_shm_dir(
                s_latlon=s_latlon,
                slm_latlon=slm_latlon,
                wall_latlon=wall_latlon,
                r_latlon=r_latlon_inside_polygon,
                Lp=spl_values,
                f=FREQUENCIES,
                T=temperature,
                alpha=source_direction,
            )
        except ValueError as e:
            logging.error(f"Could not create map: {e}")
            return

        for weighting, topic in zip(WEIGHTINGS, self.mqtt_shm_topics):

            # apply weightings
            wfunc = weightfunc(weighting[0])
            shm_weighted = np.round(shm, 1) + wfunc(FREQUENCIES.astype(float))[:, None]

            # sum or not
            if weighting.endswith("fullband"):
                # compute a fullband spectrum
                band_frequencies = ["fullband"]

                # outside the polygon everything is NaN
                L_all = np.empty((len(band_frequencies), r_latlon_all.shape[1]), dtype=object)

                # sum over rms pressures and convert back
                rms = np.sum(10 ** (shm_weighted / 10), axis=0)[None]
                shm_weighted = 10 * np.log10(rms)
            else:
                # compute 1/3 octave wise
                band_frequencies = FREQUENCIES.tolist()
                L_all = np.empty((len(band_frequencies), r_latlon_all.shape[1]), dtype=object)

            # prepare output
            L_all.fill(None)
            L_all[:, mask_inside_polygon] = shm_weighted
            L_all_final_shape = L_all.reshape((-1, grid_shape[0], grid_shape[1]))

            msg = {
                "phenomenonTime": endtime.isoformat(sep="T", timespec="seconds") + "Z",
                "resultTime": datetime.utcnow().isoformat() + "Z",
                "result": {
                    "starttime": starttime.isoformat(sep="T", timespec="seconds"),
                    "endtime": endtime.isoformat(sep="T", timespec="seconds"),
                    "timeStamp": endtime.isoformat(sep="T", timespec="seconds"),
                    "lat_0": r_latlon_all[0, 0],
                    "lon_0": r_latlon_all[1, 0],
                    "nfreq": len(band_frequencies),
                    "nrow": L_all_final_shape.shape[1],
                    "ncols": L_all_final_shape.shape[2],
                    "cellsize": cellsize,
                    "data": L_all_final_shape.tolist(),
                    "bandFrequencies": band_frequencies,
                    "unit": f"SPL dB {weighting}",
                    "input_positions": {
                        "sources_latlon": s_latlon.tolist(),
                        "slm_latlon": slm_latlon.tolist(),
                        "walls_latlon": wall_latlon.tolist()
                        if wall_latlon is not None
                        else [],
                        "shm_area": polygon.tolist(),
                    },
                },
            }

            await self.mqtt_client.publish(topic, json.dumps(msg).encode())
            logging.info(f"Published map on {topic}")

    async def start(self):
        while True:
            await self.make_map()
            await asyncio.sleep(NEW_MAP_PERIODE)


def register_shm_as_service(catalogue_url):
    """Register service at OGC service catalogue.

    Returns MQTT topics for the three SHM variations and MQTT broker address and port
    """
    topics = []
    for weighting in WEIGHTINGS:
        msg = {
            "externalId": f"SoundHeatMap/{SHM_ID}/{weighting}",
            "metadata": "SoundHeatMapGenerator",
            "sensorType": f"SoundHeatMap/L{weighting}eq",
            "unitOfMeasurement": f"Sound Heat Map with {weighting}-weighted values.",
            "fixedLatitude": 0,
            "fixedLongitude": 0,
        }
        r = requests.post(catalogue_url + "/SearchOrCreateOGCDataStreamId", json=msg)
        r.raise_for_status()

        # get MQTT topic and server
        data = r.json()

        topic = data["mqttTopic"]
        mqtt_address, mqtt_port = data["mqttServer"].split(":")
        mqtt_address = "mqtt://" + mqtt_address

        logging.info(
            f"Registered {weighting} weighting at topic {topic}, {mqtt_address}:{mqtt_port}"
        )

        topics.append(topic)

    return topics, mqtt_address, mqtt_port


async def main():
    parser = optparse.OptionParser()
    parser.add_option("-l", "--logging-level", help="Logging level")
    parser.add_option("-f", "--logging-file", help="Logging file name")
    parser.add_option("-a", "--account", help="account")
    parser.add_option("-p", "--password", help="password")
    (options, args) = parser.parse_args()

    if options.account and options.password:
        # auth data provided in command line
        auth = (options.account, options.password)
    else:
        # auth data provided from environment variables
        auth = AUTH

    # initialize logging
    logging_levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    logging_level = logging_levels.get(options.logging_level, logging.INFO)
    logging.basicConfig(
        level=logging_level,
        filename=options.logging_file,
        format="%(asctime)s %(levelname)-6s %(name)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%d-%m-%Y:%H:%M:%S",
    )

    logging.getLogger("pyost").addHandler(logging.StreamHandler())
    logging.getLogger("pyost").setLevel(logging_level)
    logging.getLogger(__name__).setLevel(logging_level)
    coloredlogs.install(level=logging_level, logger=logging.getLogger(__name__))

    mqtt_topics, mqtt_address, mqtt_port = register_shm_as_service(
        os.environ["CATALOG_URL"]
    )

    while True:
        try:
            # Connect to MQTT broker
            mqtt_client = MQTTClient()
            retcode = await mqtt_client.connect(mqtt_address, mqtt_port)
            if retcode == 0:
                logging.info(f"Connected to MQTT broker: {mqtt_address}:{mqtt_port}")
            else:
                raise ConnectionError(
                    f"Could not connect to broker, CONNACK code {retcode}"
                )

            # Connect to OGC SensorThing store server
            datastream_manager = DatastreamManager(
                GOST_URL, mqtt_client, FILTER_MICS, auth=auth
            )
            data_collector = DataCollector(mqtt_client, datastream_manager.datastreams)
            map_maker = MapMaker(
                datastream_manager.datastreams, mqtt_client, mqtt_topics
            )

            # run concurrently
            await asyncio.gather(
                datastream_manager.start(), data_collector.start(), map_maker.start()
            )
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt. Shutting down ...")
            sys.exit(0)
        except ImportError:
            logging.error(traceback.format_exc())
            sys.exit(1)
        except ConnectionError:
            logging.error(traceback.format_exc())
            logging.error("Restarting...")
        except Exception:
            logging.error(traceback.format_exc())
            logging.error("Restarting...")
        finally:
            mqtt_client.disconnect()
            break


if __name__ == "__main__":
    asyncio.run(main())
