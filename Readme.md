# Custom Model Example
You can use custom advanced tools to bring your own AI model to the Picterra platform. 
This example code shows how this might be accomplished.

You may find it useful to read the Picterra Custom Advanced Tools authoring guide first.

# Building the Image

To build the docker image and tag it with the name `picterra-byom-example`:

`docker build . -t picterra-byom-example`

# Running the Image

To run locally you should place a valid geotiff file named `raster.tif` into the `input` directory,
then run:

`docker run -v $PWD/input:/input/ -v $PWD/output:/output picterra-byom-example`

You should expect to see results.geojson appear in your `output` directory

# Publishing to the Picterra platform

While this feature is in development, please contact the <a href=mailto:support@picterra.ch>Picterra customer success team</a>
for assistance publishing your model.