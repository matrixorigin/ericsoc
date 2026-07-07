use taxi;

DROP TABLE if exists tnp22;

-- BUG #25498: external table cannot handle POINT yet, so the POINT type 
-- is loaded as varchar(200) for now.
CREATE TABLE tnp22 (
    trip_id varchar(100) not null,
    trip_start_timestamp timestamp,
    trip_end_timestamp timestamp,
    trip_seconds int,
    trip_miles float,
    pickup_census_tract varchar(100),
    dropoff_census_tract varchar(100),
    pickup_community_area int,
    dropoff_community_area int,
    fare float,
    tip float,
    additional_charges float,
    trip_total float,
    shared_trip_authorized boolean,
    trips_pooled int,
    pickup_centroid_latitude float,
    pickup_centroid_longitude float,
    pickup_centroid_location varchar(200), -- SQL Point is (LONG, LAT), this is the same as chicago file
    dropoff_centroid_latitude float,
    dropoff_centroid_longitude float,
    dropoff_centroid_location varchar(200)
);

DROP TABLE if exists tnp2324;

CREATE TABLE tnp2324 (
    trip_id varchar(100) not null,
    trip_start_timestamp varchar(100),      -- MM/DD/YYYY HH:MM:SS PM, not timestamp format.  Need STR_TO_DATE(val, '%m/%d/%Y %h:%i:%s %p')
    trip_end_timestamp varchar(100),        -- Same
    trip_seconds int,
    trip_miles float,
    percent_time_chicago float,             -- 2023/2024 only
    percent_distance_chicago float,         -- 2023/2024 only
    pickup_census_tract varchar(100),
    dropoff_census_tract varchar(100),
    pickup_community_area int,
    dropoff_community_area int,
    fare float,
    tip float,
    additional_charges float,
    trip_total float,
    shared_trip_authorized boolean,
    shared_trip_match boolean,             -- 20232024 only
    trips_pooled int,
    pickup_centroid_latitude float,
    pickup_centroid_longitude float,
    pickup_centroid_location varchar(200), -- SQL Point is (LONG, LAT), this is the same as chicago file
    dropoff_centroid_latitude float,
    dropoff_centroid_longitude float,
    dropoff_centroid_location varchar(200)
);


