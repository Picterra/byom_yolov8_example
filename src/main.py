import shapely
from osgeo import gdal, osr
from shapely import Polygon
import json
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def _assert_north_up(geot) -> None:
    """Asserts that a geotransform is north-up"""
    assert np.allclose(geot[2], 0), "Non north-up image with geot=%s" % str(geot)
    assert np.allclose(geot[4], 0), "Non north-up image with geot=%s" % str(geot)


class TransformHelper(object):
    """
    Helper object to transform coordinates between raster coords and
    WGS84
    """

    @staticmethod
    def from_dataset(ds: gdal.Dataset):
        proj = ds.GetProjection()
        geot = ds.GetGeoTransform()
        return TransformHelper.from_projection_and_geot(proj, geot)

    @staticmethod
    def from_projection_and_geot(projection: str, geot):
        return TransformHelper(projection, geot)

    def __init__(self, proj: str, geot) -> None:
        img_proj = proj
        self.geot = geot
        # We only support north-up images, in particular in lnglat2xy
        _assert_north_up(self.geot)

        img_srs = osr.SpatialReference(wkt=img_proj)
        img_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        self.img_to_wgs84 = osr.CoordinateTransformation(img_srs, wgs84_srs)
        self.wgs84_to_img = osr.CoordinateTransformation(wgs84_srs, img_srs)

    def xy2lnglat(self, x: float, y: float):
        """
        Returns the lng, lat of the center of an image pixel
        """
        x += 0.5
        y += 0.5
        x_geo = self.geot[0] + x * self.geot[1] + y * self.geot[2]
        y_geo = self.geot[3] + x * self.geot[4] + y * self.geot[5]
        lng, lat, _ = self.img_to_wgs84.TransformPoint(x_geo, y_geo)
        return lng, lat

    def lnglat2xy(self, lng: float, lat: float):
        """
        Returns the pixel coordinates of the pixel containing the provided
        lng, lat
        """
        x_geo, y_geo, _ = self.wgs84_to_img.TransformPoint(lng, lat)
        # This only works if self.geot[2] == self.geot[4] == 0
        x = (x_geo - self.geot[0]) / self.geot[1]
        y = (y_geo - self.geot[3]) / self.geot[5]
        return int(x), int(y)

    def ring_lnglat2xy(self, polygon_ring):
        """
        Converts a polygon ring from latlng to image xy coordinates
        """
        coords = []
        # Cannot iterate with for (lng, lat) because there might be a third 'elevation' value
        for values in polygon_ring:
            lng = values[0]
            lat = values[1]
            x, y = self.lnglat2xy(lng, lat)
            coords.append((x, y))
        return coords

    def ring_xy2lnglat(self, polygon_ring):
        """
        Convert a polygon ring from xy to latlng
        """
        coords = []
        for values in polygon_ring:
            x = values[0]
            y = values[1]
            coords.append(self.xy2lnglat(x, y))
        return coords

    def poly_lnglat2xy(self, coordinates):
        """
        Converts a polygons defined as a list of rings from latlng to
        image xy coordinates
        """
        return [self.ring_lnglat2xy(ring) for ring in coordinates]

    def poly_xy2lnglat(self, coordinates):
        """
        Converts a polygons defined as a list of rings from image xy coordinates to
        latlng
        """
        return [self.ring_xy2lnglat(ring) for ring in coordinates]

    def xy2lnglatbounds(self, width, height):
        """
        Returns image bounding box around [0, 0, width, height] in lnglat
        """
        bounds = []
        for x, y in [[0, 0], [width, height]]:
            x_geo = self.geot[0] + x * self.geot[1] + y * self.geot[2]
            y_geo = self.geot[3] + x * self.geot[4] + y * self.geot[5]
            lng, lat, _ = self.img_to_wgs84.TransformPoint(x_geo, y_geo)
            bounds.append([lng, lat])
        return bounds


def shapely_to_coordinates(shapely_polygon):
    ext_coords = [shapely_polygon.exterior.coords[:-1]]
    interiors_coords = [interior.coords[:-1] for interior in shapely_polygon.interiors]
    coords = ext_coords + interiors_coords
    return coords


def coordinates_to_shapely(coords):
    return shapely.geometry.Polygon(coords[0], coords[1:])


class Model:
    def __init__(self, yolov8_model_path):
        self.module = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                         model_path=yolov8_model_path,
                                                         confidence_threshold=0.01,
                                                         device="cuda:0")
        self.num_classes = 10

    def predict_img(self, img):
        slice = 2048
        self.module.model.overrides["imgsz"] = slice
        img = img[:]
        results = get_sliced_prediction(
            img,
            self.module,
            slice_height=slice,
            slice_width=slice,
            postprocess_class_agnostic=False,
            perform_standard_pred=False,
            verbose=2,
        )
        polygons_per_class = {c: [] for c in range(self.num_classes)}
        ids_per_polygons_for_all_classes = {c: [] for c in range(self.num_classes)}
        for obj in results.object_prediction_list:
            cat = obj.category.id
            if cat not in polygons_per_class:
                polygons_per_class[cat] = []
                ids_per_polygons_for_all_classes[cat] = []
            polygons_per_class[cat].append(
                [np.array(x, int) for x in shapely_to_coordinates(obj.polygon)]
            )
        return polygons_per_class


def get_input_ds(filename: str) -> gdal.Dataset:
    return gdal.Open(filename)


def _poly_to_feature(p: Polygon, poly_class) -> dict:
    poly = json.loads(shapely.to_geojson(p))
    feat = {
        "type": "Feature",
        "geometry": poly,
        "properties": {"class": poly_class}
    }
    return feat


def write_output_(filename: str, polys: list[Polygon]):
    fc = {
        "type": "FeatureCollection",
        "features": [_poly_to_feature(p) for p in polys]
    }
    with open(filename, "w") as output_fh:
        json.dump(fc, output_fh)


def write_output(filename: str, polys):
    fc = {
        "type": "FeatureCollection",
        "features": [_poly_to_feature(p, class_poly) for p, class_poly in polys]
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
    model = Model("../weight/best.pt")
    img = raster_ds.ReadAsArray().transpose(1, 2, 0)[:, :, :3]
    res = model.predict_img(img)
    helper = TransformHelper.from_dataset(raster_ds)
    res2 = {k: [coordinates_to_shapely(helper.poly_xy2lnglat(x)) for x in v] for k, v in res.items()}
    res3 = []
    for k, v in res2.items():
        for vv in v:
            res3.append((vv, k))
    return res3


if __name__ == '__main__':
    input_filename = '/input/raster.tif'
    output_filename = '/output/results.geojson'
    raster_ds = get_input_ds(input_filename)
    geom_colleciton = my_model(raster_ds=raster_ds)
    write_output(output_filename, geom_colleciton)
