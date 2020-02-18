# Monica sound heat map module

This module is providing a web service for computation of sound heat maps as part of the MONICA cloud solutions. The module continuously collects observation from sound pressure level measurements and outputs sound heat maps via MQTT. The maps are estimates of sound pressure levels over a specified area around the measurements using a simple sound propagation model. The model is based on acoustic free-field mono- or dipoles.

## Deployment

This module is configured with a set of environment variables, see [wiki](https://github.com/MONICA-Project/sound-heat-map/wiki/Configuration-via-environment-variables). One may either change them in the [Dockerfile](https://github.com/MONICA-Project/sound-heat-map/blob/master/Dockerfile) and rebuild the image before running the container

```bash
# change Dockerfile ...
docker build -t some_tag .
docker run -t some_tag
```

or set the environment files dynamically when running the container with the [`-e, --env, --env-file` options](https://docs.docker.com/engine/reference/commandline/run/#set-environment-variables--e---env---env-file).

A Docker image can also be found at [MONICA's dockerhub repository](https://hub.docker.com/repository/docker/monicaproject/sound-heat-map)

## Development

### Run locally
Run locally without Docker using the `run_local.py` script.

```bash
# create virtual environment, e.g. with conda
conda create -n shm python pip
conda activate shm
# install dependencies
pip install -r requirements.txt
# run locally
cd app
python run_local.py --account XXXX --password XXXX --logging_level [info, debug, warning ...]
```

### Testing
There are two Python scripts in the `tools` folder that might be handy while setting up this module.

- `tools/check_positions.py` visualizes positional configuartion parameters like the sound heat map area and source positions on a google map widget. Use this to verify your positons before deployment.
- `tools/display_maps.py` listens to sound heat maps published over the MQTT broker and displays them in the browser. Use this to verify and inspect the module output without running a [COP](https://github.com/MONICA-Project/COP-UI) instance.  

## Contributing
Contributions are welcome. 

Please fork, make your changes, and submit a pull request. For major changes, please open an issue first and discuss it with the other authors.

## Affiliation
![MONICA](https://github.com/MONICA-Project/template/raw/master/monica.png)  
This work is supported by the European Commission through the [MONICA H2020 PROJECT](https://www.monica-project.eu) under grant agreement No 732350.
