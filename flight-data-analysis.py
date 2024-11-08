from pyspark.sql import SparkSession
from pyspark.sql.functions import abs, col, rank, when, to_timestamp, stddev, count, round, avg, expr
from pyspark.sql.window import Window

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------

def task1_largest_discrepancy(flights_df, airports_df, carriers_df):
    flights_df = flights_df.withColumn("scheduled_travel_time", col("ScheduledArrival") - col("ScheduledDeparture")) \
                           .withColumn("actual_travel_time", col("ActualArrival") - col("ActualDeparture")) \
                           .withColumn("discrepancy", abs(col("scheduled_travel_time") - col("actual_travel_time")))

    window = Window.partitionBy("CarrierCode").orderBy(col("discrepancy").desc())
    result_df = flights_df.withColumn("rank", rank().over(window)) \
                          .filter(col("rank") <= 3) \
                          .select("FlightNum", "CarrierCode", "Origin", "Destination", "scheduled_travel_time", "actual_travel_time", "discrepancy")

    result_df = result_df.join(carriers_df, result_df.CarrierCode == carriers_df.CarrierCode, how="left") \
                         .join(airports_df.alias("origin_airport"), result_df.Origin == col("origin_airport.AirportCode"), how="left") \
                         .join(airports_df.alias("dest_airport"), result_df.Destination == col("dest_airport.AirportCode"), how="left") \
                         .select("FlightNum", "CarrierName", 
                                 col("origin_airport.AirportName").alias("OriginAirportName"), col("origin_airport.City").alias("OriginCity"),
                                 col("dest_airport.AirportName").alias("DestAirportName"), col("dest_airport.City").alias("DestCity"),
                                 "scheduled_travel_time", "actual_travel_time", "discrepancy")

    result_df.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------

def task2_consistent_airlines(flights_df, carriers_df):
    flights_df = flights_df.withColumn("departure_delay_seconds", expr("CAST(UNIX_TIMESTAMP(ActualDeparture) - UNIX_TIMESTAMP(ScheduledDeparture) AS DOUBLE)"))

    result_df = flights_df.groupBy("CarrierCode") \
                          .agg(stddev("departure_delay_seconds").alias("std_dev"), count("*").alias("num_flights")) \
                          .filter(col("num_flights") > 100) \
                          .join(carriers_df, "CarrierCode", how="left") \
                          .orderBy(col("std_dev"))

    result_df.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------

def task3_canceled_routes(flights_df, airports_df):
    # Mark canceled flights
    flights_df = flights_df.withColumn("is_canceled", when(col("ActualDeparture").isNull(), 1).otherwise(0))

    # Group by route and calculate cancellation rate
    route_stats = flights_df.groupBy("Origin", "Destination") \
                           .agg(count("*").alias("total_flights"), 
                                count(when(col("is_canceled") == 1, True)).alias("canceled_flights"))

    result_df = route_stats.withColumn("cancellation_rate", round(col("canceled_flights") / col("total_flights"), 2)) \
                           .orderBy(col("cancellation_rate").desc())

    # Join with airport information for route origin and destination
    result_df = result_df.join(airports_df.alias("origin_airport"), col("Origin") == col("origin_airport.AirportCode"), how="left") \
                         .join(airports_df.alias("dest_airport"), col("Destination") == col("dest_airport.AirportCode"), how="left") \
                         .select(
                             col("origin_airport.AirportName").alias("OriginAirportName"), 
                             col("origin_airport.City").alias("OriginCity"), 
                             col("dest_airport.AirportName").alias("DestAirportName"), 
                             col("dest_airport.City").alias("DestCity"), 
                             "cancellation_rate"
                         )

    # Write the result to a CSV file
    result_df.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------

def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # Categorize flights based on time of day
    flights_df = flights_df.withColumn(
        "time_of_day",
        when((col("ScheduledDeparture") >= "06:00:00") & (col("ScheduledDeparture") < "12:00:00"), "morning")
        .when((col("ScheduledDeparture") >= "12:00:00") & (col("ScheduledDeparture") < "18:00:00"), "afternoon")
        .when((col("ScheduledDeparture") >= "18:00:00") | (col("ScheduledDeparture") < "06:00:00"), "evening")
        .otherwise("night")
    )

    # Calculate average delay for each carrier and time of day
    result_df = flights_df.withColumn("departure_delay", col("ActualDeparture") - col("ScheduledDeparture")) \
                          .groupBy("CarrierCode", "time_of_day") \
                          .agg(avg("departure_delay").alias("avg_delay")) \
                          .join(carriers_df, on="CarrierCode", how="left") \
                          .select("CarrierName", "time_of_day", "avg_delay")

    # Write the result to a CSV file
    result_df.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")


# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, airports_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
