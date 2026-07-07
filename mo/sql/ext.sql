use taxi;

DROP TABLE tnp_2022_100k;

-- BUG #25498: external table cannot handle POINT yet, so the POINT type 
-- is loaded as varchar(200) for now.
CREATE EXTERNAL TABLE tnp_2022_100k (
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
)
INFILE 'stage://fengttt_public/chicago-taxi/tnp_2022_100k.csv'
IGNORE 1 LINES;


