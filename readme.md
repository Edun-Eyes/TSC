## Usage

How to run the code:

Start an experiment by:

* ``run_Hypergrapg.py``

  Run the pipeline under different traffic flows. Specific traffic flow files as well as basic configuration can be assigned in this file. For details about config, please turn to ``config.py``.


For most cases, you might only modify traffic files and config parameters in ``runexp.py``.



## Agent

* `Hmasac_agent.py`



## Others

More details about this project are demonstrated in this part.

* ``config.py``

  The whole configuration of this project. Note that some parameters will be replaced in ``runexp.py`` while others can only be changed in this file, please be very careful!!!

* ``pipeline.py``

  The whole pipeline is implemented in this module:

  Start a simulator environment, run a simulation for certain time(one round), construct samples from raw log data, update the model and model pooling.

* ``generator.py``

  A generator to load a model, start a simulator enviroment, conduct a simulation and log the results.

* ``anon_env.py``

  Define a simulator environment to interact with the simulator and obtain needed data like features.

* ``construct_sample.py``

* Construct training samples from original data. Select desired state features in the config and compute the corrsponding average/instant reward with specific measure time.

* ``updater.py``

  Define a class of updater for model updating.

