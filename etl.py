from pyspark.sql.functions import monotonically_increasing_id as mon_id
from pyspark.sql.functions import udf, col, lit, year, month, upper, to_date
from pyspark.sql.types import DateType, StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.sql import SparkSession
import configparser
import os
import sys
import logging

import pandas as pd

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = "/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/jvm/java-8-openjdk-amd64/bin"
os.environ["SPARK_HOME"] = "/opt/spark-2.4.3-bin-hadoop2.7"
os.environ["HADOOP_HOME"] = "/opt/spark-2.4.3-bin-hadoop2.7"


# setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# AWS configuration
config = configparser.ConfigParser()
config.read('variables.cfg', encoding='utf-8-sig')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']
SOURCE_S3_BUCKET = config['S3']['SOURCE_S3_BUCKET']
DEST_S3_BUCKET = config['S3']['DEST_S3_BUCKET']


# ETL functions

# create spark session
def create_spark_session():
    """
    Description: Creates spark session.

    Returns:
        spark session object
    """

    try:
        AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
        AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

        spark = SparkSession.builder\
            .config("spark.jars.packages",
                    "saurfang:spark-sas7bdat:2.0.0-s_2.11")\
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0")\
            .enableHiveSupport().getOrCreate()
        logger.info("Spark session created")
        return spark
    except Exception as e:
        logger.error("Error creating spark session: {}".format(e))
        raise e


def SAS_to_date(date):
    if date is not None:
        return pd.to_timedelta(date, unit='D') + pd.Timestamp('1960-1-1')


SAS_to_date_udf = udf(SAS_to_date, DateType())


# rename columns
def rename_columns(table, new_column_names):
    for original, new in zip(table.columns, new_column_names):
        table = table.withColumnRenamed(original, new)
    return table


# process immigration data
def process_immigration_data(spark, output_data):
    """
    Process immigration data to get fact and dimension tables

        Args:
            spark {object}: SparkSession object
            input_data {object}: Source S3 endpoint
            output_data {object}: Destination S3 endpoint

        Returns:
            None
    """

    logging.info("Processing immigration data starting...")

    # read immigration data
    immigration_data = "18-83510-I94-Data-2016/*.sas7bdat"
    df = spark.read.format("com.github.saurfang.sas.spark").load(
        immigration_data, forceLowercaseNames=True, inferLong=True)

    logging.info("Processing fact tables")

    # extract columns to create fact tables
    fact_immigration = df.select('cicid', 'i94yr', 'i94mon', 'i94port', 'i94addr',
                                 'arrdate', 'depdate', 'i94mode', 'i94visa').distinct()\
        .withColumn("immigration_id", mon_id())

    # data wrangling
    new_column_names = ['cic_id', 'year', 'month', 'city_code', 'state_code',
                        'arrive_date', 'departure_date', 'mode', 'visa']

    fact_immigration = rename_columns(fact_immigration, new_column_names)

    fact_immigration = fact_immigration.withColumn(
        "country", lit("United States"))
    fact_immigration = fact_immigration.withColumn(
        "arrive_date", SAS_to_date_udf(col("arrive_date")))
    fact_immigration = fact_immigration.withColumn(
        "departure_date", SAS_to_date_udf(col("departure_date")))

    # write fact_immigration table to parquet file partitioned by state and city
    fact_immigration.write.partitionBy('state_code', 'city_code').parquet(
        os.path.join(output_data + "fact_immigration/"), mode='overwrite')

    logging.info("Processing dim_person_immigration tables")

    # extract columns  to create dim_person_immigration table
    dim_person_immigration = df.select("cicid", "i94cit", "i94res", "biryear", "gender", "insnum").distinct()\
        .withColumn("dim_person_immigration_id", mon_id())

    # data wrangling
    new_column_names = ["cic_id", "citizen_country",
                        "residence_country", "birth_year", "gender", "ins_num"]
    dim_person_immigration = rename_columns(
        dim_person_immigration, new_column_names)

    # write dimension table to parquet file
    dim_person_immigration.write.mode("overwrite").parquet(
        path=output_data + "dim_person_immigration")

    logging.info("Processing dimension airline table")

    # extract columns to create dim_airline_immigration table
    dim_airline_immigration = df.select("cicid", "airline", "admnum", "fltno", "visatype").distinct()\
                                .withColumn("airline_immigration_id", mon_id())

    # data wrangling
    new_column_names = ["cic_id", "airline",
                        "admin_num", "flight_number", "visa_type"]
    dim_airline_immigration = rename_columns(
        dim_airline_immigration, new_column_names)

    # write dim_airline_immigration table to parquet file
    dim_airline_immigration.write.mode("overwrite").parquet(
        path=output_data + "dim_airline_immigration")


def process_label_description_data(spark, output_data):
    """ Parsing label desctiption file to get codes of country, city, state

        Arguments:
            spark {object}: SparkSession object
            input_data {object}: Source S3 endpoint
            output_data {object}: Target S3 endpoint

        Returns:
            None
    """

    logging.info("Processing label description data starting...")

    # read label description data
    label = "I94_SAS_Labels_Descriptions.SAS"
    with open(label, "r") as f:
        contents = f.readlines()

        country_codes = {}
        for countries in contents[10:298]:
            pair = countries.split('=')
            code, country = pair[0].strip(), pair[1].strip().strip("'")
            country_codes[code] = country
        spark.createDataFrame(country_codes.items(), ["code", "country"]).write.mode(
            "overwrite").parquet(path=output_data + "country_codes")

        city_code = {}
        for cities in contents[303:962]:
            pair = cities.split('=')
            code, city = pair[0].strip("\t").strip().strip("'"), \
                pair[1].strip("\t").strip().strip("'")
            city_code[code] = city
        spark.createDataFrame(city_code.items(), ["code", "city"]).write.mode(
            "overwrite").parquet(path=output_data + "city_code")

        state_code = {}
        for states in contents[982:1036]:
            pair = states.strip("=")
            code, state = pair[0].strip("\t").strip().strip("'"), \
                pair[1].strip("\t").strip().strip("'")
            state_code[code] = state
        spark.createDataFrame(state_code.items(), ["code", "state"]).write.mode(
            "overwrite").parquet(path=output_data + "state_code")


def process_temperature_data(spark, output_data):
    """ Process temperature data to get dim_temperature table

        Arguments:
            spark {object}: SparkSession object
            input_data {object}: Source S3 endpoint
            output_data {object}: Target S3 endpoint

        Returns:
            None
    """

    logging.info("Processing temperature data starting...")

    # read temperature data
    temperature_data = "GlobalLandTemperaturesByCity.csv"
    df = spark.read.format("csv").option(
        "header", "true").load(temperature_data)

    # extract columns to create dim_temperature table
    df = df.where(df["country"] == "United States")
    dim_temperature = df.select(
        ["dt", "AverageTemperature", "AverageTemperatureUncertainty", "City", "Country"]).distinct()

    # data wrangling
    new_column_names = ["dt", "average_temperature",
                        "average_temperature_uncertainty", "city", "country"]
    dim_temperature = rename_columns(dim_temperature, new_column_names)

    dim_temperature = dim_temperature.withColumn('dt', to_date(col('dt')))
    dim_temperature = dim_temperature.withColumn(
        'year', year(dim_temperature['dt']))
    dim_temperature = dim_temperature.withColumn(
        'month', month(dim_temperature['dt']))

    # write dim_temperature table to parquet file partitioned by year and month
    dim_temperature.write.mode("overwrite").parquet(
        path=output_data + "dim_temperature")


def process_demography_data(spark, output_data):
    """ Process demography data to get dim_demography table

        Arguments:
            spark {object}: SparkSession object
            input_data {object}: Source S3 endpoint
            output_data {object}: Target S3 endpoint

        Returns:
            None
    """

    logging.info("Processing demography data starting...")

    # read demography data
    demography_data = "us-cities-demographics.csv"
    df = spark.read.format("csv").option(
        "header", "true").load(demography_data)

    # extract columns to create dim_demography table
    dim_demography = df.select(
        ["City", "State", "Male Population", "Female Population", "Number of Veterans", "Foreign-born", "Race"]).distinct()

    # data wrangling
    new_column_names = ["city", "state", "female_population", "female_population",
                        "num_of_veterans", "foreign_born", "race"]
    dim_demography = rename_columns(dim_demography, new_column_names)

    # write dim_demography table to parquet file
    dim_demography.write.mode("overwrite").parquet(
        path=output_data + "dim_demography")

    logging.info("Processing dim_demography_stat starting...")

    dim_demography_stat = df.select(["City", "State", "Median Age", "Average Household Size"]).distinct()\
                            .withColumn("demography_stat_id", mon_id())

    # data wrangling
    new_column_names = ["city", "state",
                        "median_age", "average_household_size"]
    dim_demography_stat = rename_columns(dim_demography_stat, new_column_names)
    dim_demog_statistics = dim_demog_statistics.withColumn(
        'city', upper(col('city')))
    dim_demog_statistics = dim_demog_statistics.withColumn(
        'state', upper(col('state')))

    # write dim_demography_stat table to parquet file
    dim_demography_stat.write.mode("overwrite").parquet(
        path=output_data + "dim_demography_stat")


def main():
    spark = create_spark_session()
    output_data = "s3a://udacity-spark-proj/output/"
    # output_data = DEST_S3_BUCKET

    process_immigration_data(spark, output_data)
    process_label_description_data(spark, output_data)
    process_temperature_data(spark, output_data)
    process_demography_data(spark, output_data)

    logging.info("Processing data finished.")


if __name__ == "__main__":
    main()
