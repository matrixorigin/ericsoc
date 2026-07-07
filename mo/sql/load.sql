use taxi;

LOAD DATA INFILE 'stage://fengttt_public/chicago-taxi/tnp_2022_100k.csv'
INTO TABLE tnp22
IGNORE 1 LINES;

LOAD DATA INFILE 'stage://fengttt_public/chicago-taxi/tnp_2023_2024_100k.csv'
INTO TABLE tnp2324
IGNORE 1 LINES;

