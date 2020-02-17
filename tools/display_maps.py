"""Listen to sound heat maps and display in browser using bokeh"""

import argparse
import asyncio
import atexit
import json
import logging
import webbrowser
from dataclasses import dataclass
import os
import sys

import numpy as np
from beeprint import pp
from bokeh.io import output_file, save
from bokeh.models import GMapOptions
from bokeh.plotting import gmap
from hbmqtt.client import ClientException, MQTTClient
from hbmqtt.mqtt.constants import QOS_0
from matplotlib import cm
from matplotlib.colors import to_hex


@dataclass
class MapDisplayer:
    """Listens to SHM messages and displays them."""

    broker_address: str
    broker_port: int
    topic: str
    freqindx: int

    async def start(self):
        # connect to MQTT broker
        self.mqtt_client = MQTTClient()

        retcode = await self.mqtt_client.connect(self.broker_address, self.broker_port)
        if retcode == 0:
            logging.info(
                f"Connected to MQTT broker: {self.broker_address}:{self.broker_port}"
            )
        else:
            raise ConnectionError(
                f"Could not connect to broker, CONNACK code {retcode}"
            )

        # subscribe to SHM topic
        code = await self.mqtt_client.subscribe([(self.topic, QOS_0)])

        if code[0] not in [0, 1, 2]:
            logging.error(
                f"""SUBACK return code {code}. Could not subscribe to
                topic: {self.topic}"""
            )
        else:
            logging.info(f"Subscribed to topic: {self.topic}")

        timeout = 60

        while True:

            try:
                message = await self.mqtt_client.deliver_message(timeout=timeout)
                logging.info(f"Received message on {message.topic}")
                logging.debug(f"Payload: {pp(json.loads(message.data), output=False)}")

                self.process(message.data)

            except ClientException as ce:
                logging.error("Client exception: %s" % ce)
            except asyncio.TimeoutError:
                logging.error(
                    f"Timeout error: did not receive message since {timeout}s"
                )

    def process(self, data):
        """Plot one SHM data."""
        obs = json.loads(data)["result"]

        nrow = obs["nrow"]
        ncol = obs["ncols"]
        lat_0 = obs["lat_0"]
        lon_0 = obs["lon_0"]
        dl = obs["cellsize"]
        data = np.array(obs["data"])

        # create positions of datapoints
        lat = lat_0 + np.arange(ncol) * dl
        lon = lon_0 + np.arange(nrow) * dl
        x, y = np.meshgrid(lat, lon)
        r_latlon = np.stack((x.flatten(), y.flatten()))

        slm_latlon = np.array(obs["input_positions"]["slm_latlon"])
        source_latlon = np.array(obs["input_positions"]["sources_latlon"])
        wall_latlon = np.array(obs["input_positions"]["walls_latlon"]).T
        shm_area = np.array(obs["input_positions"]["shm_area"])

        map_pos = source_latlon[:, 0]
        # plot
        output_file("gmap.html")
        map_options = GMapOptions(
            lat=map_pos[0], lng=map_pos[1], zoom=18, map_type="roadmap", tilt=0
        )
        p = gmap(
            " AIzaSyCd60D3jnN2Ehyhprg1yiYMjQKp-dKfiu0 ",
            map_options,
            title="Tivoli",
            height=700,
            width=1200,
        )

        allvalues = data[self.freqindx].flatten()
        values = allvalues[allvalues != None].astype(float)
        normed_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        colors = np.empty(allvalues.shape, dtype=object)
        colors.fill("#000000")
        colors.fill(None)
        colors[allvalues != None] = [to_hex(c) for c in cm.viridis(normed_values)]

        p.circle_cross(
            r_latlon[1, :][allvalues != None],
            r_latlon[0, :][allvalues != None],
            fill_color=colors[allvalues != None],
            size=10,
        )
        p.circle_cross(
            slm_latlon[1, :],
            slm_latlon[0, :],
            legend="SLM",
            size=15,
            fill_color="green",
        )
        # p.circle_cross(r_latlon[1, mask], r_latlon[0, mask], color = 'w', edgecolor = 'k')
        if wall_latlon.size > 0:
            p.multi_line(
                list(wall_latlon[:, :, 1]), list(wall_latlon[:, :, 0]), legend="Walls"
            )
        # p.plot(area[1, :, :], area[0, :, :], c = 'k')
        p.circle_cross(
            source_latlon[1, :], source_latlon[0, :], legend="sources", size=15
        )
        p.circle_cross(
            shm_area[1, :],
            shm_area[0, :],
            legend="SHM area",
            size=15,
            fill_color="yellow",
        )
        p.circle_cross(lon_0, lat_0, legend="reference", size=15, fill_color="black")

        save(p)

        webbrowser.open("gmap.html")
        print("Plotted.")


async def main():
    parser = argparse.ArgumentParser(
        description="""
        Display sound heat maps in the browser using bokeh.
        Reads content of Dockerfile for sound heat map parameters.
        """
    )
    parser.add_argument(
        "mqtt_address", help="MQTT broker address, e.g. mqtt://127.0.0.1"
    )
    parser.add_argument(
        "-p", "--mqtt-port", help="MQTT broker port, default 1883", default=1883
    )
    parser.add_argument(
        "topic",
        help="MQTT topic, e.g. 'MQTT_BROKER_PREFIX/Datastreams(1)/Observations'",
    )
    parser.add_argument(
        "-f", "--freq-index", help="frequency index, default 0", default=0
    )
    args = parser.parse_args()
    print(args)
    md = MapDisplayer(
        args.mqtt_address, int(args.mqtt_port), args.topic, int(args.freq_index),
    )

    # run concurrently
    await asyncio.gather(md.start())


if __name__ == "__main__":

    def cleanup():
        os.chdir(sys.path[0])
        os.remove("gmap.html")

    atexit.register(cleanup)
    asyncio.run(main())
