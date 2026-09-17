/**
 * GEE Code Editor script
 * 生成 2021-11 至 2022-05 的 7 个 S1/S2 月合成，以及 1 个 77 波段时序影像。
 * 此历史导出脚本声明的单月波段顺序（尚未独立核实全部归档波段语义）：
 * B2, B3, B4, B8, NDVI, EVI, GNDVI, YCI, SAVI, VV, VH
 *
 * 使用方法：复制到 https://code.earthengine.google.com/ 运行，然后在 Tasks 中
 * 点击 8 个 Run。文件会输出到 Google Drive/S1S2_Fusion_2021_2022。
 */

// 现有 GeoTIFF 的精确外包矩形。若你在 GEE 中有原始遂宁矢量边界，建议将下一行
// 替换为：var roi = ee.FeatureCollection('projects/你的项目/assets/边界').geometry();
var roi = ee.Geometry.Rectangle(
  [103.90067289724686, 30.67836528695842,
   104.32575568889245, 30.96708380028179],
  'EPSG:4326', false
);

var outputFolder = 'S1S2_Fusion_2021_2022';
var outputCrs = 'EPSG:4326';
// 与现有文件的像元大小 0.000089831528412° 完全一致。
var outputTransform = [
  0.000089831528412, 0, 103.90067289724686,
  0, -0.000089831528412, 30.96708380028179
];

// Sentinel-2 SR：按 SCL 去除无效像元、云、云影和雪，并转换为 0~1 反射率。
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
  // 与现有影像像元值一致：YCI = (green - red) / (green + red)。
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

// 把 7 个月按时间顺序堆叠。给每月波段添加 YYYYMM 后缀，避免 GEE 自动产生 _1、_2。
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
