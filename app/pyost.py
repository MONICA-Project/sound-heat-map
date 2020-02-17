"""OGC SensorThing client.

Simple API wrapper to query for objects in a OGC SensorThingStore.
"""

import logging
import requests
from urllib.parse import urljoin
import os

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class SensorThingStore:
    """A SensorThing Object Store client."""

    def __init__(self, url, auth=None):
        self.baseurl = url  # no trailing backslash!
        self.auth = auth
        # get SensorThings resource endpoint
        linklist = self.get_or_fail(self.baseurl)["value"]
        self.resources = {d["name"]: d["url"] for d in linklist}

    def get_or_fail(self, url):
        r = requests.get(url, auth=self.auth)

        if r.status_code == 404:
            return None

        r.raise_for_status()
        return r.json()

    def get_by_id(self, kind, id):
        url = urljoin(self.resources[kind], "{0}({1})".format(kind, id))

        return self.get_or_fail(url)

    def get_by_query(self, kind, id=None, **query_kwargs):
        id_str = "({})".format(id) if id is not None else ""
        query_str = "?" + ";".join(
            ["${0}={1}".format(key, value) for key, value in query_kwargs.items()]
        )
        url = self.resources[kind] + id_str + query_str

        return self.get_or_fail(url)["value"]

    def get_by_parent(self, kind, parent, id, **query_kwargs):
        id_str = "({})".format(id)
        query_str = "?" + ";".join(
            ["${0}={1}".format(key, value) for key, value in query_kwargs.items()]
        )
        url = self.resources[parent] + id_str + "/" + kind + query_str

        return self.get_or_fail(url)["value"]


# for testing
if __name__ == "__main__":
    gost_url = "http://SOMEURL/v1.0"

    store = SensorThingStore(gost_url)

    print("RESOURCES")
    print(store.resources)

    print("GET BY ID")
    print(store.get_by_id("Things", 1))

    print("GET BY QUERY")
    print(store.get_by_query("Things", filter="equals(name, 'QUERY')"))

    print("GET BY PARENT")
    print(store.get_by_parent("Locations", "Things", 1, expand="HistoricalLocations"))
