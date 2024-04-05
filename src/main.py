import shapely
from osgeo import gdal, osr
from shapely import Polygon
import json


def get_input_ds(filename: str) -> gdal.Dataset:
    return gdal.Open(filename)


def _poly_to_feature(p: Polygon) -> dict:
    poly = json.loads(shapely.to_geojson(p))
    feat = {
        "type": "Feature",
        "geometry": poly,
        "properties": {}
    }
    return feat


def write_output(filename: str, polys: list[Polygon]):
    fc = {
        "type": "FeatureCollection",
        "features": [_poly_to_feature(p) for p in polys]
    }
    with open(filename, "w") as output_fh:
        json.dump(fc, output_fh)


def my_model(raster_ds: gdal.Dataset) -> list[Polygon]:
    """
    You can replace this with calling your custom model code.

    This example code generates a polygon with dimensions approx. 10% of the size of the raster
    with its origin in the center of a given raster

    :param raster_ds: a gdal dataset containing the raster data
    :return: a shapely GeometryCollection
    """
    # Create polygon naively using pixel coordinates
    x = raster_ds.RasterXSize / 2
    y = raster_ds.RasterYSize / 2
    size_x = int(raster_ds.RasterXSize / 10)
    size_y = int(raster_ds.RasterYSize / 10)
    coords = ((x, y), (x, y + size_y), (x + size_x, y + size_y), (x + size_x, y))
    # transform the polygon to the georeference of the image
    # See https://gdal.org/tutorials/geotransforms_tut.html
    projection = raster_ds.GetProjection()
    geot = raster_ds.GetGeoTransform()
    img_srs = osr.SpatialReference(wkt=projection)
    img_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geo_coords = []
    for x, y in coords:
        x += 0.5
        y += 0.5
        x_geo = geot[0] + x * geot[1] + y * geot[2]
        y_geo = geot[3] + x * geot[4] + y * geot[5]
        geo_coords.append((x_geo, y_geo))
    # And then to 4326 (Picterra standardises on this)
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wgs84_geot = osr.CoordinateTransformation(img_srs, wgs84_srs)
    wgs84_coords = []
    for x, y in geo_coords:
        lng, lat, _ = wgs84_geot.TransformPoint(x, y)
        wgs84_coords.append((lng, lat))
    poly = shapely.Polygon(wgs84_coords)
    return [poly, ]


if __name__ == '__main__':
    input_filename = '/input/raster.tif'
    output_filename = '/output/results.geojson'
    raster_ds = get_input_ds(input_filename)
    geom_colleciton = my_model(raster_ds=raster_ds)
    write_output(output_filename, geom_colleciton)
