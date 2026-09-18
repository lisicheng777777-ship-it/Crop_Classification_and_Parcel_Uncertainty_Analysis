/**
 * GEE Code Editor script
 * Generate seven monthly S1/S2 composites for November 2021 through May 2022 and one 77-band time-series image.
 * Monthly band order declared by this historical script (archived band semantics are not fully verified):
 * B2, B3, B4, B8, NDVI, EVI, GNDVI, YCI, SAVI, VV, VH
 *
 * Usage: paste into https://code.earthengine.google.com/ and run, then start
 * all eight export tasks in Tasks. Outputs are saved to Google Drive/S1S2_Fusion_2021_2022.
 */

// Exact bounding rectangle of the existing GeoTIFF. If the original Suining boundary is available in GEE,
// replace the next line with: var roi = ee.FeatureCollection('projects/YOUR_PROJECT/assets/BOUNDARY').geometry();
var roi = ee.Geometry.Rectangle(
  [103.90067289724686, 30.67836528695842,
   104.32575568889245, 30.96708380028179],
  'EPSG:4326', false
);

var outputFolder = 'S1S2_Fusion_2021_2022';
var outputCrs = 'EPSG:4326';
// Match the existing pixel size of 0.000089831528412 degrees.
var outputTransform = [
  0.000089831528412, 0, 103.90067289724686,
  0, -0.000089831528412, 30.96708380028179
];

// Sentinel-2 SR: mask invalid pixels, clouds, cloud shadows and snow using SCL; scale reflectance to 0-1.
function maskS2(image) {
  var scl = image.select('SCL');
  var good = scl.neq(0)   // No data
    .and(scl.neq(1))      // Saturated/defective
    .and(scl.neq(3))      // Cloud shadow
    .and(scl.neq(8))      // Cloud, medium probability
    .and(scl.neq(9))      // Cloud, high probability
    .and(scl.neq(10))     // Thin cirrus
    .and(scl.neq(11));    // Snow/ice
  return image.select(['B2', 'B3', 'B4', 'B8'])
    .multiply(0.0001)
    .updateMask(good)
    .copyProperties(image, ['system:time_start']);
}

function addIndices(optical) {
  var b2 = optical.select('B2');
  var b3 = optical.select('B3');
  var b4 = optical.select('B4');
  var b8 = optical.select('B8');

  var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI');
  var evi = b8.subtract(b4).multiply(2.5)
    .divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
    .rename('EVI');
  var gndvi = b8.subtract(b3).divide(b8.add(b3)).rename('GNDVI');
  // Historical YCI formula: (green - red) / (green + red).
  var yci = b3.subtract(b4).divide(b3.add(b4)).rename('YCI');
  var savi = b8.subtract(b4).multiply(1.5)
    .divide(b8.add(b4).add(0.5)).rename('SAVI');

  return optical.addBands([ndvi, evi, gndvi, yci, savi]);
}

function monthlyFusion(startText) {
  var start = ee.Date(startText);
  var end = start.advance(1, 'month');

  var optical = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(start, end)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 80))
    .map(maskS2)
    .median()
    .clip(roi);
  optical = addIndices(optical);

  var radar = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH'])
    .median()
    .clip(roi);

  return optical.addBands(radar)
    .select(['B2', 'B3', 'B4', 'B8', 'NDVI', 'EVI', 'GNDVI', 'YCI',
             'SAVI', 'VV', 'VH'])
    .toFloat()
    .set('system:time_start', start.millis());
}

var months = [
  ['2021-11-01', '2021_11', '202111'],
  ['2021-12-01', '2021_12', '202112'],
  ['2022-01-01', '2022_01', '202201'],
  ['2022-02-01', '2022_02', '202202'],
  ['2022-03-01', '2022_03', '202203'],
  ['2022-04-01', '2022_04', '202204'],
  ['2022-05-01', '2022_05', '202205']
];

var monthlyImages = [];
months.forEach(function(item) {
  var image = monthlyFusion(item[0]);
  monthlyImages.push(image);

  Export.image.toDrive({
    image: image,
    description: 'Fusion_' + item[1],
    folder: outputFolder,
    fileNamePrefix: 'Fusion_' + item[1],
    region: roi,
    crs: outputCrs,
    crsTransform: outputTransform,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });
});

// Stack seven months chronologically. Add YYYYMM suffixes to avoid automatic _1 and _2 suffixes.
var baseBandNames = ['B2', 'B3', 'B4', 'B8', 'NDVI', 'EVI', 'GNDVI',
                     'YCI', 'SAVI', 'VV', 'VH'];
var timeStack = ee.Image([]);
monthlyImages.forEach(function(image, i) {
  var suffix = months[i][2];
  var datedNames = baseBandNames.map(function(name) {
    return name + '_' + suffix;
  });
  timeStack = timeStack.addBands(image.rename(datedNames));
});

Export.image.toDrive({
  image: timeStack.toFloat(),
  description: 'Fusion_202111_202205',
  folder: outputFolder,
  fileNamePrefix: 'Fusion_202111_202205',
  region: roi,
  crs: outputCrs,
  crsTransform: outputTransform,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

Map.centerObject(roi, 10);
Map.addLayer(monthlyImages[0], {bands: ['B8', 'B4', 'B3'], min: 0, max: 0.4},
             '2021-11 false color');
print('Single-month bands', monthlyImages[0].bandNames());
print('Time-stack bands', timeStack.bandNames());
