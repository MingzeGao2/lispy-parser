import gdal
from gdalconst import *
import osr
import numpy as np

def read_raster(input_file_name, band_index=1):
    dataset = gdal.Open(input_file_name, GA_ReadOnly)
    input_driver_name = dataset.GetDriver().ShortName
    band = dataset.GetRasterBand(band_index)
    nodata = band.GetNoDataValue()
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    data = band.ReadAsArray(0, 0, x_size, y_size)
    geotransform = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    return (data, x_size, y_size, nodata, input_driver_name, geotransform, proj)

def create_raster(x_size, y_size, output_file_name, 
                  input_driver_name, georef, proj):
    """Create the output raster file
    
    Keyword argument:
    x_size              -- size in x dimension for output raster band
    y_size              -- size in y dimension for output raster band
    output_file_name    -- file name for output data
    input_driver_name   -- output file format
    georef              -- geo reference from input data
    proj                -- projection from input data

    Return value:
    output_dataset      -- output data set 
    """
    driver = gdal.GetDriverByName(input_driver_name)
    output_dataset = driver.Create(output_file_name, x_size, 
                                   y_size, 1, gdal.GDT_Float32)
    output_dataset.SetGeoTransform(georef)
    output_dataset.SetProjection(proj)
    return output_dataset
        
    
def write_raster(dataset, data, x_offset, 
                 y_offset, nodata):
    """ Write data to one bands of output file 
            
    Keyword arguments:
    dataset             -- output data set
    data                -- data to be write, numpy array 
    x_offset            -- x offset for each band 
    y_offset            -- y offset for each band
    nodata              -- nodata value for each band
    """
    dataset.GetRasterBand(1).WriteArray(data, x_offset, y_offset)
    dataset.GetRasterBand(1).SetNoDataValue(nodata)
    
