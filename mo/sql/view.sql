use taxi;

DROP VIEW if exists tnp22_v;
DROP VIEW if exists tnp2324_v;

-- BUG #25498: external table cannot handle POINT yet, so the POINT type 
-- is loaded as varchar(200) for now.
CREATE VIEW tnp22_v as
    SELECT trip_id, 
    start_ts as start_ts,
    year(start_ts) as start_year, 
    month(start_ts) as start_month,
    dayofyear(start_ts) as start_doy,
    dayofweek(start_ts) as start_dow,    -- sunday:1, monday:2, ..., saturday:7
    weekday(start_ts) as start_wd,   -- monday:0, ........, saturday:5, sunday:6
    hour(start_ts) as start_hour,
    minute(start_ts) as start_minute,
    end_ts as end_ts,
    year(end_ts) as end_year, 
    month(end_ts) as end_month,
    dayofyear(end_ts) as end_doy,
    dayofweek(end_ts) as end_dow,
    weekday(end_ts) as end_wd,  
    hour(end_ts) as end_hour,
    minute(end_ts) as end_minute,
    trip_seconds,
    trip_miles,
    pickup_census_tract, dropoff_census_tract,
    pickup_community_area, dropoff_community_area,
    fare, tip, additional_charges, trip_total,
    shared_trip_authorized, trips_pooled,
    pickup_centroid_latitude, pickup_centroid_longitude, pickup_centroid_location,
    dropoff_centroid_latitude, dropoff_centroid_longitude, dropoff_centroid_location
    FROM (
    SELECT trip_id, 
    trip_start_timestamp as start_ts,
    trip_end_timestamp as end_ts,
    trip_seconds,
    trip_miles,
    pickup_census_tract, dropoff_census_tract,
    pickup_community_area, dropoff_community_area,
    fare, tip, additional_charges, trip_total,
    shared_trip_authorized, trips_pooled,
    pickup_centroid_latitude, pickup_centroid_longitude, pickup_centroid_location,
    dropoff_centroid_latitude, dropoff_centroid_longitude, dropoff_centroid_location
    FROM tnp22
    ) subt;

CREATE VIEW tnp2324_v as
    SELECT trip_id, 
    start_ts as start_ts,
    year(start_ts) as start_year, 
    month(start_ts) as start_month,
    dayofyear(start_ts) as start_doy,
    dayofweek(start_ts) as start_dow,
    weekday(start_ts) as start_wd,  
    hour(start_ts) as start_hour,
    minute(start_ts) as start_minute,
    end_ts as end_ts,
    year(end_ts) as end_year, 
    month(end_ts) as end_month,
    dayofyear(end_ts) as end_doy,
    dayofweek(end_ts) as end_dow,
    weekday(end_ts) as end_wd, 
    hour(end_ts) as end_hour,
    minute(end_ts) as end_minute,
    trip_seconds,
    trip_miles,
    pickup_census_tract, dropoff_census_tract,
    pickup_community_area, dropoff_community_area,
    fare, tip, additional_charges, trip_total,
    shared_trip_authorized, trips_pooled,
    pickup_centroid_latitude, pickup_centroid_longitude, pickup_centroid_location,
    dropoff_centroid_latitude, dropoff_centroid_longitude, dropoff_centroid_location
    FROM (
    SELECT trip_id, 
    str_to_date(trip_start_timestamp, '%m/%d/%Y %h:%i:%s %p')::timestamp as start_ts, 
    str_to_date(trip_end_timestamp, '%m/%d/%Y %h:%i:%s %p')::timestamp as end_ts, 
    trip_seconds,
    trip_miles,
    pickup_census_tract, dropoff_census_tract,
    pickup_community_area, dropoff_community_area,
    fare, tip, additional_charges, trip_total,
    shared_trip_authorized, trips_pooled,
    pickup_centroid_latitude, pickup_centroid_longitude, pickup_centroid_location,
    dropoff_centroid_latitude, dropoff_centroid_longitude, dropoff_centroid_location
    FROM tnp2324
    ) subt;

