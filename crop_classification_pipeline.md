# A Comprehensive Approach to Crop Classification Using Earth Observation Technologies and Machine Learning

This document outlines a Python script designed for comprehensive crop classification using Earth observation data and machine learning techniques. The script encompasses several key stages, including:

* **Data Acquisition:** Fetching satellite imagery from Google Earth Engine and auxiliary geospatial data.
* **Preprocessing:** Cleaning and preparing the raw data for analysis.
* **Feature Engineering:** Deriving meaningful features from the preprocessed data.
* **Model Training:** Constructing and training a machine learning model for classification.
* **Evaluation:** Assessing the performance of the trained model.

The script is structured into logical blocks, each with specific responsibilities and aims to be self-explanatory through the comments.

## Initial Setup and Configuration

This section initializes the Earth Engine API, defines essential paths, and imports necessary libraries.

```python
import ee
ee.Authenticate()
ee.Initialize()

# Define base folder and file paths for data storage
base_folder = 'crops/'
base_labels = f'{base_folder}/labels-polygons'
output_tiff_path = f'{base_labels}/etiquetas-2017-2018.tif'
outfile = f'{base_folder}/features_yaqui_crops.csv'
# Import necessary libraries
import numpy as np   
import geopandas as gpd 
import tarfile 
import glob 
import subprocess 
from datetime import datetime  
import rsgislib  
import rasterio
import rasterio.features
from rsgislib import rastergis
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os
```

## Creating the Base Directory

This block creates the base directory for storing all data if it doesn't exist.

```python
# --- 2. Create Directory ---
# Creates base directory if it does not exist
base_folder = 'crops/'
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
    print(f"Directory '{base_folder}' created successfully.")
else:
    print(f"Directory '{base_folder}' already exists.")
```

## Downloading and Unzipping the Area of Interest (AOI) Shapefile

This part of the script downloads a zip file containing the AOI shapefile and extracts its contents.

```python
# --- 3. Download AOI Shapefile and Unzip ---
import os
import requests
import zipfile

# Define the directory where the file will be downloaded
base_folder = 'crops/'
zip_file_path = os.path.join(base_folder, 'aoi_yaqui_son.zip')

# Create the directory if it doesn't exist
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

# URL of the file to download
url = 'https://github.com/abxda/crop-classification-tutorial/raw/refs/heads/main/crops/aoi_yaqui_son.zip'

# Download the file
print("Downloading the file...")
response = requests.get(url)
with open(zip_file_path, 'wb') as file:
    file.write(response.content)
print(f"File downloaded: {zip_file_path}")

# Unzipping the file
print("Unzipping the file...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(base_folder)
print(f"File unzipped to: {base_folder}")

# Optional: Remove the zip file after extraction
os.remove(zip_file_path)
print(f"Zip file removed: {zip_file_path}")
# Read the shapefile
shapefile_path = f'{base_folder}aoi_yaqui_son.shp'

gdf = gpd.read_file(shapefile_path)
# Convert shapefile to geojson for use with google earth engine.
region_geojson = gdf.geometry[0].__geo_interface__
# Convert the geojson to an ee.Geometry object
study_area = ee.Geometry(region_geojson)
```

## Defining Earth Engine Functions

This section defines several key functions for processing imagery in Google Earth Engine. These include:

* `s2mask`: Masks clouds in Sentinel-2 imagery.
* `addVariables`: Calculates spectral indices.
* `get_image_collection`: Retrieves and processes Sentinel-2 imagery.

```python
# --- 4. Define Earth Engine Functions ---
# Function to mask clouds in Sentinel-2 imagery
def s2mask(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).int16()

# Function to add spectral indices as bands
def addVariables(image):
    return image \
        .addBands(image.normalizedDifference(['nir', 'red']).multiply(10000).int16().rename('NDVI')) \
        .addBands(image.expression('2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))', {
            'nir': image.select('nir'),
            'red': image.select('red'),
            'blue': image.select('blue')
        }).multiply(10000).int16().rename('EVI')) \
        .addBands(image.expression('nir / green - 1', {
            'nir': image.select('nir'),
            'green': image.select('green')
        }).multiply(10000).int16().rename('GCVI')) \
        .addBands(image.expression('1 / 2 * (2 * nir + 1 - ((2 * nir + 1) ** 2 - 8 * (nir - red)) ** (1 / 2))', {
            'nir': image.select('nir'),
            'red': image.select('red')
        }).multiply(10000).int16().rename('MSAVI2')) \
        .addBands(image.normalizedDifference(['nir', 'swir1']).multiply(10000).int16().rename('LSWI')) \
        .addBands(image.normalizedDifference(['swir1', 'red']).multiply(10000).int16().rename('NDSVI')) \
        .addBands(image.normalizedDifference(['swir1', 'swir2']).multiply(10000).int16().rename('NDTI'))

# Function to retrieve and process Sentinel-2 imagery
def get_image_collection(start_date, end_date):

    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(study_area) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(s2mask) \
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']) \
        .map(addVariables)

    merged = s2

    num_bands = merged.first().bandNames().length()
    geometric_median = merged.reduce(ee.Reducer.geometricMedian(num_bands)).toInt16()

    return geometric_median
```

## Splitting the Study Area into Smaller Regions

This section defines a function to split the study area into smaller regions for efficient processing.

```python
# --- 5. Split Study Area into Smaller Regions for Download ---
import math

def split_geometry(geometry, max_dim=0.2):
    bounds = geometry.bounds().getInfo()['coordinates'][0]
    minX, minY = bounds[0]
    maxX, maxY = bounds[2]
    width = maxX - minX
    height = maxY - minY

    x_steps = math.ceil(width / max_dim)
    y_steps = math.ceil(height / max_dim)

    x_step_size = width / x_steps
    y_step_size = height / y_steps

    regions = []
    for i in range(x_steps):
        for j in range(y_steps):
            region = ee.Geometry.Rectangle(
                [minX + i * x_step_size,
                 minY + j * y_step_size,
                 minX + (i + 1) * x_step_size,
                 minY + (j + 1) * y_step_size])
            regions.append(region)
    return regions
```

## Downloading and Compressing Images

This section defines functions to download and compress satellite imagery in GeoTIFF format.
These include:

* `download_image`: Download a single image.
* `download_images`: Download images for all regions.
* `compress_tiff_directory`: Compresses a directory of tiff to tar.gz.
* `create_unified_tif`: Merge a collection of tiff images in a single one.
* `delete_files_by_pattern`: Delete files by pattern.

```python
# --- 6. Download and Compress Images ---
import os
import requests
import shutil

# Function to download a single image
def download_image(image, region, scale, base_filename):   
    if os.path.exists(base_filename):
        print(f"File {base_filename} already exists. Skipping download.")
        return  

    url = image.getDownloadURL({
        'region': region,
        'scale': scale,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    })


    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(base_filename, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
    else:
        print(f'Error downloading {base_filename}: {response.status_code}')

# Function to download images for all the regions
def download_images(image, regions, scale, folder):
    os.makedirs(folder, exist_ok=True)
    for i, region in enumerate(regions):
        filename = os.path.join(folder, f'image_{i}.tif')
        download_image(image, region.getInfo()['coordinates'], scale, filename)

# Function to compress a directory of TIFF files into a tar archive
def compress_tiff_directory(tiff_directory, tar_filename):
    with tarfile.open(tar_filename, 'w') as tar:
        for root, dirs, files in os.walk(tiff_directory):
            for file in files:
                if file.endswith('.tif'):
                    file_path = os.path.join(root, file)
                    tar.add(file_path, arcname=os.path.relpath(file_path, tiff_directory))

# Function to merge tiff images
def create_unified_tif(input_pattern, output_tif, compression='LZW'):
    tif_files = glob.glob(input_pattern)

    if not tif_files:
        print("No TIFF files conforming to the specified pattern were located.")
        return
    merge_command = [
        'gdal_merge',
        '-o', output_tif,
        '-of', 'GTiff',
        '-co', f'COMPRESS={compression}'
    ] + tif_files
    subprocess.run(merge_command)
    print(f"Imagen unificada creada: {output_tif}")
# Function to delete a collection of files
def delete_files_by_pattern(pattern):
  files = glob.glob(pattern)
  for file in files:
    try:
      os.remove(file)
      #print(f"Deleted file: {file}")
    except OSError as e:
      print(f"Error deleting file {file}: {e}")
# Define date ranges for data retrieval
date_ranges = [
    ('2018-02-01', '2018-02-28'),
    ('2018-03-01', '2018-03-31'),
    ('2018-04-01', '2018-04-30')
]
# Split study area
regions = split_geometry(study_area)
```

## Create a GeoPackage File to Store Regions

This part of the script takes the regions generated in the previous step and creates a GeoPackage file to store them

```python
# --- 7. Create GeoPackage file to store regions ---
from shapely.geometry import Polygon
import pandas as pd
gdf_list = []
for region in regions:
    coords = region.coordinates().getInfo()[0]
    polygon = Polygon(coords)
    gdf = gpd.GeoDataFrame({'geometry': [polygon]})
    gdf_list.append(gdf)

gdf_final = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
gdf_final = gdf_final.set_crs(epsg=4326) 
gdf_final.to_file(f'{base_folder}gee_regions.gpkg', driver='GPKG', index=False)
```

## Downloading Sentinel-2 Images

This section downloads Sentinel-2 images for each defined date range and stores them in a compressed format.

```python
# --- 8. Download Sentinel-2 Images for each date range ---
for start_date, end_date in date_ranges:
    geometric_median_image = get_image_collection(start_date, end_date)
    folder_name = f'img-multispectral-{start_date[:7]}'
    output_folder = os.path.join(base_folder, folder_name)
    download_images(geometric_median_image, regions, 30, output_folder)
    tar_filename = os.path.join(base_folder, f'{folder_name}.tar')
    compress_tiff_directory(output_folder, tar_filename)    
    print(f"Images {start_date[:7]} downloaded and compressed {tar_filename}")
```

## Create Unified Tiff Files for Each Month of Sentinel-2

Here, the script merges all the downloaded Sentinel-2 images for each month in a single unified Tiff

```python
# --- 9. Create a unified tiff files for each month of Sentinel-2  ---
create_unified_tif('crops/img-multispectral-2018-02/*.tif', 'crops/img-multispectral-2018-02/img-optica-2018-02.tif')
create_unified_tif('crops/img-multispectral-2018-03/*.tif', 'crops/img-multispectral-2018-03/img-optica-2018-03.tif')
create_unified_tif('crops/img-multispectral-2018-04/*.tif', 'crops/img-multispectral-2018-04/img-optica-2018-04.tif')
```

## Delete Individual Tiff Files

This block deletes the temporal individual Tiff files created in the previous step

```python
# --- 10. Delete individual tiff files  ---
delete_files_by_pattern('crops/img-multispectral-2018-01/image_*.tif')
```

## Defining Sentinel-1 Processing Functions

This section defines functions to process Sentinel-1 data, including:

* `calculate_rvi`: Calculates the Radar Vegetation Index.
* `get_sentinel1_data`: Retrieves Sentinel-1 data.
* `get_monthly_medians`: Calculates monthly medians.
* `download_image_collection_by_month`: Downloads Sentinel-1 images on a monthly basis.

```python
# --- 11. Define Sentinel-1 processing functions ---
import math
import os
import requests
import shutil

def calculate_rvi(image):
    vv = image.select('VV')
    vh = image.select('VH')
    q = vh.divide(vv)
    N = q.multiply(q.add(3))
    D = q.add(1).pow(2)
    rvi = N.divide(D).rename('RVI')
    return image.addBands(rvi)

# Function to retrieve Sentinel-1 data
def get_sentinel1_data(study_area, start_date, end_date):
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterDate(start_date, end_date) \
        .filterBounds(study_area) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .select(['VV', 'VH']) \
        .map(calculate_rvi)
    return s1

# Function to get monthly medians of an image collection
def get_monthly_medians(image_collection):
    def monthly_median(month):
        month = ee.Number(month).int()
        year = ee.Date(image_collection.first().get('system:time_start')).get('year')
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
        month_col = image_collection.filterDate(start_date, end_date)
        median = month_col.median().set('month', month).set('year', year)
        band_names = median.bandNames()
        return ee.Algorithms.If(band_names.size(), median, None)

    months = ee.List.sequence(1, 12)
    monthly_medians = ee.ImageCollection.fromImages(months.map(monthly_median).removeAll([None]))
    return monthly_medians

# Function to download image collection by month
def download_image_collection_by_month(image_collection, regions, scale, folder, prefix='Sentinel1_RVI'):
    os.makedirs(folder, exist_ok=True)
    images = image_collection.toList(image_collection.size())
    num_images = images.size().getInfo()

    for i in range(num_images):
        img = ee.Image(images.get(i))
        month = img.get('month').getInfo()
        year = img.get('year').getInfo()

        for j, region in enumerate(regions):
            filename = os.path.join(folder, f'{prefix}_{year}_{month}_region_{j}.tif')
            download_image(img, region.getInfo()['coordinates'], scale, filename)
```

## Downloading and Compressing Sentinel-1 Images

This section downloads and compresses the Sentinel-1 data on a monthly basis.

```python
# --- 12. Download and Compress Sentinel-1 Images ---
date_ranges = [
    ('2018-02-01', '2018-02-28'),
    ('2018-03-01', '2018-03-31'),
    ('2018-04-01', '2018-04-30')
]

for start_date, end_date in date_ranges:
    sentinel1_data = get_sentinel1_data(study_area, start_date, end_date)   
    monthly_medians = get_monthly_medians(sentinel1_data)
    folder_name = f'img-radar-{start_date[:7]}'
    output_folder = os.path.join(base_folder, folder_name)

    download_image_collection_by_month(monthly_medians, regions, 30, output_folder)

    tar_filename = os.path.join(base_folder, f'{folder_name}.tar')
    compress_tiff_directory(output_folder, tar_filename)
    create_unified_tif(f'{output_folder}/*.tif', f'{output_folder}/{folder_name}.tif')
    delete_files_by_pattern(f'{output_folder}/Sentinel1*.tif')

    print(f"Imágenes de {start_date[:7]} descargadas y comprimidas en {tar_filename}")
```

## Download Geometric Median Image for the Entire Period

This part downloads a geometric median image of Sentinel-2 for the entire period.

```python
# --- 13. Download Geometric Median Image of Sentinel-2 for the entire period ---
start_date = '2018-02-01'
end_date = '2018-04-30'
geometric_median_image = get_image_collection(start_date, end_date)
folder_name = f'img-multispectral-GM-{start_date[:4]}'
output_folder = os.path.join(base_folder, folder_name)
download_images(geometric_median_image, regions, 30, output_folder)
tar_filename = os.path.join(base_folder, f'{folder_name}.tar')
compress_tiff_directory(output_folder, tar_filename) 
gm_tif = f'{output_folder}/{folder_name}.tif'
create_unified_tif(f'{output_folder}/*.tif', gm_tif)
delete_files_by_pattern(f'{output_folder}/image_*.tif')
```

## Image Segmentation

This section performs image segmentation using the rsgislib library.

```python
# --- 14. Image Segmentation ---
from rsgislib.segmentation import shepherdseg
from rsgislib.vectorutils import createvectors

inputImage = f'{output_folder}/{folder_name}.tif'
clumpsImage = f'{output_folder}/{folder_name}-k-80-d-100.kea'
outShp = f'{output_folder}/{folder_name}-k-80-d-100.shp'
tmpDir  = f'{output_folder}/rsgislibsegtmp-all'

shepherdseg.run_shepherd_segmentation(inputImage,
                                      clumpsImage,
                                      tmp_dir=tmpDir,
                                      num_clusters=80,
                                      min_n_pxls=100,
                                      bands=[1,2,3,4,5,6,7,8,9,10,11,12,13],
                                      dist_thres=100,
                                      sampling=100,
                                      km_max_iter=200,
                                      process_in_mem=True)

createvectors.polygonise_raster_to_vec_lyr(input_img=clumpsImage, 
                                           out_vec_file=outShp, 
                                           out_vec_lyr="clusters", 
                                           out_format="ESRI Shapefile")
```

## Download Labels GeoPackage

This part downloads the labels of the different crops in a geopackage file

```python
# --- 15. Download Labels GeoPackage ---
geopackage_file_path = os.path.join(base_folder, 'crop_labels_yaqui.gpkg')
# URL of the file to download
url = 'https://github.com/abxda/crop-classification-tutorial/raw/refs/heads/main/crops/crop_labels_yaqui.gpkg'
# Download the file
print("Downloading the file...")
response = requests.get(url)
with open(geopackage_file_path, 'wb') as file:
    file.write(response.content)
print(f"File downloaded: {geopackage_file_path}")
# Read the labels
samples = gpd.read_file(geopackage_file_path)
```

## Intersect Polygons with Labels

This section intersects the segmented polygons with the downloaded labels to associate each segment with a crop type.

```python
# --- 16.  Intersect Polygons with Labels  ---
polygons = gpd.read_file(outShp)

import geopandas as gpd
# Perform a spatial join to find intersecting points for each polygon
# This avoids explicit loops and is more efficient
joined = gpd.sjoin(polygons, samples[['name', 'geometry']], how='left', predicate='intersects')
# Group the joined data by the index of the polygons and aggregate unique 'name's
grouped = joined.groupby(joined.index).agg({
    'name': lambda x: x.dropna().unique().tolist(),
    'geometry': 'first'  # Keep the original geometry of the polygons
})
# Convert the grouped data back into a GeoDataFrame
results = gpd.GeoDataFrame(grouped, geometry='geometry')
# Define the maximum number of labels (etiquetas) you expect
max_labels = 6
# Expand the list of 'name's into separate columns 'etiqueta_1', 'etiqueta_2', etc.
for i in range(max_labels):
    results[f'etiqueta_{i+1}'] = results['name'].apply(lambda x: x[i] if i < len(x) else None)
# List of etiqueta columns for convenience
etiqueta_columns = [f'etiqueta_{i+1}' for i in range(max_labels)]
# Drop rows where all etiqueta columns are NaN (i.e., no intersecting points)
results = results.dropna(subset=etiqueta_columns, how='all')
# Filter polygons that have exactly one etiqueta (label)
filtered_results = results[results['name'].apply(lambda x: len(x) == 1)]
```

## Save Clean Polygons with Labels as GeoPackage

This block saves the filtered polygons with associated crop labels into a GeoPackage file.

```python
# --- 17. Save clean polygons with labels as geopackage ---
shapefile_path = os.path.join(base_folder, 'clean_polygons.gpkg')

filtered_results = filtered_results.set_crs("EPSG:4326")
filtered_results.to_file(shapefile_path)
```

## Create an Integer Classification for Labels

This part converts text labels into a numeric format for machine learning purposes and saves the mapping.

```python
# --- 18. Create an integer classification for labels ---
clases_unicas = filtered_results['etiqueta_1'].unique()
clase_a_numero = {clase: num for num, clase in enumerate(clases_unicas, start=1)}
filtered_results['Clase_Numerada'] = filtered_results['etiqueta_1'].map(clase_a_numero)
correspondencia_df = pd.DataFrame(list(clase_a_numero.items()), columns=['Clase_Textual', 'Clase_Numerada'])
output_shapefile_path = os.path.join(base_folder, 'clean_polygons_int_class.gpkg')
filtered_results.to_file(output_shapefile_path)
equivalence_csv = os.path.join(base_folder, 'equivalence_yaqui_crops.csv')
correspondencia_df.to_csv(equivalence_csv, index=False)
```

## Create a TIFF with the Labels from the Shapefile

This block generates a rasterized version of the labeled polygons for further analysis.

```python
# --- 19. Create a Tiff with the labels from the shapefile ---
gdf = gpd.read_file(output_shapefile_path)
gdf.set_crs(epsg=4326, inplace=True)

with rasterio.open(gm_tif) as ref_tiff:
    ref_transform = ref_tiff.transform
    ref_crs = ref_tiff.crs
    ref_width = ref_tiff.width
    ref_height = ref_tiff.height

shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.Clase_Numerada))
raster = np.zeros((ref_height, ref_width), dtype=np.int8)
burned = rasterio.features.rasterize(shapes=shapes, out_shape=raster.shape, transform=ref_transform, fill=-1)

# Guardar el raster en un archivo TIFF
raster_meta = {
    'driver': 'GTiff',
    'height': ref_height,
    'width': ref_width,
    'count': 1,
    'dtype': 'float32',
    'crs': ref_crs,
    'transform': ref_transform,
    'nodata': -1
}

output_tiff_path = os.path.join(base_folder, 'labels-2017-2018.tif')
output_tiff_path

with rasterio.open(output_tiff_path, 'w', **raster_meta) as dst:
    dst.write(burned, 1)
```

## Backup and Restore Functions for KEA Files

These functions provide backup and restore capabilities for KEA files, enhancing data safety.

```python
# --- 20. Backup Function for .kea files ---
def backup_kea_file(clumpsImage, backup_dir):
    """
    Crea un respaldo del archivo KEA en la carpeta de respaldo especificada, agregando un timestamp al nombre del archivo.

    :param clumpsImage: Ruta del archivo KEA original.
    :param backup_dir: Directorio de respaldo donde se guardará la copia.
    """
    # Asegurarse de que el directorio de respaldo existe
    os.makedirs(backup_dir, exist_ok=True)

    # Obtener el nombre del archivo original sin la ruta
    base_name = os.path.basename(clumpsImage)

    # Generar un timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Crear el nuevo nombre del archivo con el timestamp
    backup_name = f"{base_name.replace('.kea', '')}_{timestamp}.kea"

    # Ruta completa del archivo de respaldo
    backup_path = os.path.join(backup_dir, backup_name)

    # Hacer una copia del archivo KEA
    shutil.copy2(clumpsImage, backup_path)

    print(f"Respaldo creado: {backup_path}")

# --- 21. Restore Function for .kea files ---
def restore_latest_kea(backup_dir, original_path):
    """
    Restaura el archivo KEA más reciente desde el directorio de respaldo a su ubicación original,
    eliminando el archivo original existente.

    :param backup_dir: Directorio de respaldo donde se encuentran los archivos KEA.
    :param original_path: Ruta original del archivo KEA que se desea restaurar.
    """
    # Verificar si el directorio de respaldo existe y contiene archivos
    if not os.path.exists(backup_dir):
        print(f"El directorio de respaldo {backup_dir} no existe.")
        return

    # Obtener la lista de archivos KEA en el directorio de respaldo
    backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.kea')]

    if not backup_files:
        print(f"No se encontraron archivos KEA en el directorio de respaldo {backup_dir}.")
        return

    # Encontrar el archivo KEA más reciente
    latest_file = max(backup_files, key=lambda f: os.path.getmtime(os.path.join(backup_dir, f)))
    latest_file_path = os.path.join(backup_dir, latest_file)

    # Eliminar el archivo original si existe
    if os.path.exists(original_path):
        os.remove(original_path)
        print(f"Archivo original eliminado: {original_path}")

    # Restaurar el archivo KEA más reciente
    shutil.copy2(latest_file_path, original_path)
    print(f"Archivo restaurado desde {latest_file_path} a {original_path}")
```

## Back Up KEA File Before Processing

This section calls the backup function to create a backup of the KEA file.

```python
# --- 22. Backup .kea file before processing ---
kea_rsp = os.path.join(output_folder, 'kea_rsp')

backup_kea_file(clumpsImage,kea_rsp)
```

## Add Class Proportion to the RAT Table

This part calculates and adds class proportions to the RAT (Raster Attribute Table).

```python
# --- 23. Add Class Proportion to the RAT Table ---
rastergis.populate_rat_with_cat_proportions(output_tiff_path, clumpsImage, out_cols_name='klass_', maj_col_name='klass')
```

## Back Up the Modified KEA File

This block backs up the modified KEA file after adding the class proportions.

```python
# --- 24. Backup the modified .kea file ---
backup_kea_file(clumpsImage,kea_rsp)
```

## Functions to Add Statistics for Bands into RAT Table

These functions calculate and add statistical information for each band to the RAT table.

```python
# --- 25. Functions to Add Statistics for bands into RAT Table ---
def agregar_estadisticas(input_tiff, clumps_kea, offset, num_bandas):
    bs = []
    for i in range(1, num_bandas + 1):
        min_field = f'b{offset + i}Min'
        max_field = f'b{offset + i}Max'
        mean_field = f'b{offset + i}Mean'
        sum_field = f'b{offset + i}Sum'
        std_dev_field = f'b{offset + i}StdDev'
        bs.append(rastergis.BandAttStats(band=i, min_field=min_field, max_field=max_field, 
                                         mean_field=mean_field, sum_field=sum_field, 
                                         std_dev_field=std_dev_field))

    rastergis.populate_rat_with_stats(input_tiff, clumps_kea, bs)

def agregar_estadisticas_multiple(base_path, clumps_kea, filenames, initial_offset, num_bandas):
    offset = initial_offset
    for filename in filenames:
        input_tiff = f"{base_path}/{filename}"    
        agregar_estadisticas(input_tiff, clumps_kea, offset, num_bandas)
        offset += num_bandas
```

## Add Statistics for Optical Images into RAT Table

This part applies the statistic calculation to the optical imagery data and adds the result to the RAT table.

```python
# --- 26. Add Statistics for Optical Images into RAT Table ---
initial_offset = 0
num_bandas = 13


date_ranges = [
    ('2018-02-01', '2018-02-28'),
    ('2018-03-01', '2018-03-31'),
    ('2018-04-01', '2018-04-30')
]

filenames = [
'img-multispectral-GM-2018/img-multispectral-GM-2018.tif',
'img-multispectral-2018-02/img-optica-2018-02.tif',
'img-multispectral-2018-03/img-optica-2018-03.tif',
'img-multispectral-2018-04/img-optica-2018-04.tif',
]

agregar_estadisticas_multiple(base_folder, clumpsImage, filenames, initial_offset, num_bandas)
```

## Add Statistics for Radar Images into RAT Table

This section applies the same statistical calculations to the radar imagery.

```python
# --- 27. Add Statistics for Radar Images into RAT Table ---
# Aplicación de la función a los archivos de imágenes de Sentinel-1
# Anteriormente se procesaron 4 imagenes x 13 bandas
initial_offset = 53
num_bandas = 3

filenames = [
'img-radar-2018-02/img-radar-2018-02.tif',
'img-radar-2018-03/img-radar-2018-03.tif',
'img-radar-2018-04/img-radar-2018-04.tif'
]

agregar_estadisticas_multiple(base_folder, clumpsImage, filenames, initial_offset, num_bandas)
```

## Export RAT Table to CSV

This block extracts all the features generated in the previous step and stores them in a CSV.

```python
# --- 28. Export RAT table to CSV ---
from rsgislib import rastergis
band_names = rastergis.get_rat_columns(clumpsImage)

filtered_band_names = [name for name in band_names if not (name.startswith('klass__') or
                                                           name.startswith('Histogram') or
                                                           name.startswith('Red') or
                                                           name.startswith('Green') or
                                                           name.startswith('Blue') or
                                                           name.startswith('Alpha'))]

rastergis.export_rat_cols_to_ascii(clumpsImage, outfile, filtered_band_names)
```

## Functions to Process CSV Data, Map Classes, and Balance Data

This section defines functions for data loading, class mapping, balancing and splitting.

```python
# --- 29. Functions to process CSV data, map classes, and balance data ---
def cargar_datos(csv_path):
    return pd.read_csv(csv_path)

def mapear_clases(data, correspondencia_df, col_original='klass', col_map='Clase_Textual', drop_na=True):
    correspondencia_dict = dict(zip(correspondencia_df['Clase_Numerada'], correspondencia_df['Clase_Textual']))
    data[col_map] = data[col_original].map(correspondencia_dict)
    if drop_na:
        data = data.dropna(subset=[col_map])
    return data

def mapear_land_cover(data, samples, col_clase='Clase_Textual', col_land_cover='land_cover'):
    samples_dict = dict(zip(samples['name'], samples['land_cover']))
    data[col_land_cover] = data[col_clase].map(samples_dict)
    return data

def balancear_clases(data, col_clase='land_cover', max_samples=500, random_state=42):
    conteo_clases = data[col_clase].value_counts()
    clases_grandes = conteo_clases[conteo_clases > max_samples].index
    frames = []
    for clase in clases_grandes:
        clase_df = data[data[col_clase] == clase]
        clase_df_reducido = clase_df.sample(n=max_samples, random_state=random_state)
        frames.append(clase_df_reducido)
    otras_clases_df = data[~data[col_clase].isin(clases_grandes)]
    data_balanceado = pd.concat([otras_clases_df] + frames)
    return data_balanceado

def dividir_datos(data, test_size=0.3, random_state=42):
    return train_test_split(data, test_size=test_size, random_state=random_state)

def fit_and_test(train_data, testing_data, klass, pipeline, nombre_pipeline, klass_test=''):
    if klass_test == '':
        klass_test = klass

    features = train_data.drop(['FID','klass',klass,'Clase_Textual'], axis=1)
    target = train_data[klass]
    testing_features = testing_data.drop(['FID','klass',klass_test,'Clase_Textual'], axis=1)
    test_target = testing_data[klass_test]

    print("Inicia entrenamiento")
    pipeline.fit(features, target)
    print("Clasificación")
    results = pipeline.predict(testing_features)
    print(f"El resultado de la clasificación con el pipeline: {nombre_pipeline}")
    print(classification_report(test_target, results, digits=4))
    return pipeline
```

## Define Machine Learning Pipelines

This section sets up the machine learning pipelines using scikit-learn.

```python
# --- 30. Define Machine Learning Pipelines ---
# Definir pipelines
exported_pipeline = make_pipeline(
    StandardScaler(),
    StackingEstimator(estimator=MLPClassifier(alpha=0.01, learning_rate_init=0.5)),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.8, min_samples_leaf=3, min_samples_split=5, n_estimators=100)
)

exported_pipeline_feature = make_pipeline(
    StandardScaler(),
    SelectPercentile(score_func=f_classif, percentile=30),
    StackingEstimator(estimator=MLPClassifier(alpha=0.01, learning_rate_init=0.5)),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.8, min_samples_leaf=3, min_samples_split=5, n_estimators=100)
)
```

## Load Data, Preprocess, Split, Train, and Evaluate the Model

Finally, the script loads the data, preprocesses it, splits the data into training and testing sets, trains the model using the training data, and evaluates the trained model using the testing data.

```python
# --- 31. Load data, preprocess, split, train, and evaluate the model ---
data = cargar_datos(outfile)
data = mapear_clases(data, correspondencia_df)
data = mapear_land_cover(data, samples)
data_balanceado = balancear_clases(data)

train_data, test_data = dividir_datos(data_balanceado)

klass = 'land_cover'
nombre_pipeline = "Extra-Trees para Yaqui"
fit_and_test(train_data, test_data, klass, exported_pipeline, nombre_pipeline)
```

This concludes the Python script for comprehensive crop classification using Earth observation technologies and machine learning. 
