python -m src.patterns.compare_geo_segments --ref "./output/centerline/geo_1.json" --targets "./output/centerline/geo_2.json" "./output/centerline/geo_3.json"
[OK] geo_2.json
  matched_node_count    = 18
  matched_segment_count = 18
  json                  = output\compare_geo\geo_1_vs_geo_2.json
  image                 = output\compare_geo\geo_1_vs_geo_2_vis.png
    S0 -> S0   | rot=   0.3596 deg | mid_t=(   -1.50,     0.00) | len_ratio=0.9965
    S1 -> S1   | rot=  -0.0475 deg | mid_t=(   -1.50,     0.50) | len_ratio=1.0029
   S10 -> S10  | rot=  -0.0711 deg | mid_t=(   -1.00,    -1.00) | len_ratio=0.9945
   S11 -> S11  | rot=   0.0316 deg | mid_t=(   -0.50,    -1.50) | len_ratio=1.0024
   S12 -> S12  | rot=  -0.0609 deg | mid_t=(   -0.50,    -0.50) | len_ratio=1.0002
   S13 -> S13  | rot=  -0.2061 deg | mid_t=(    0.50,    -0.50) | len_ratio=0.9832
   S14 -> S14  | rot=   0.0304 deg | mid_t=(    0.50,    -0.50) | len_ratio=1.0025
   S15 -> S15  | rot=   0.0197 deg | mid_t=(   -0.50,     0.50) | len_ratio=1.0016
   S16 -> S16  | rot=  -0.2362 deg | mid_t=(    0.00,     0.50) | len_ratio=1.0061
   S17 -> S17  | rot=  -0.0383 deg | mid_t=(   -0.50,     2.00) | len_ratio=1.0052
    S2 -> S2   | rot=  -0.3548 deg | mid_t=(    1.00,    -0.50) | len_ratio=0.9889
    S3 -> S3   | rot=   0.0412 deg | mid_t=(    0.50,    -1.50) | len_ratio=0.9975
    S4 -> S4   | rot=  -0.0051 deg | mid_t=(   -0.50,    -3.00) | len_ratio=0.9978
    S5 -> S5   | rot=   0.2115 deg | mid_t=(    0.00,     0.50) | len_ratio=1.0069
    S6 -> S6   | rot=  -0.0564 deg | mid_t=(    0.00,     0.50) | len_ratio=0.9982
    S7 -> S7   | rot=   0.0420 deg | mid_t=(   -0.50,    -0.50) | len_ratio=0.9976
    S8 -> S8   | rot=  -0.0189 deg | mid_t=(   -0.50,    -0.50) | len_ratio=1.0011
    S9 -> S9   | rot=   0.0000 deg | mid_t=(   -2.00,     0.00) | len_ratio=1.0000

[OK] geo_3.json
  matched_node_count    = 18
  matched_segment_count = 18
  json                  = output\compare_geo\geo_1_vs_geo_3.json
  image                 = output\compare_geo\geo_1_vs_geo_3_vis.png
    S0 -> S9   | rot=   0.1985 deg | mid_t=(   -4.00,    -0.50) | len_ratio=1.0063
    S1 -> S10  | rot=   0.0336 deg | mid_t=(    0.00,     7.50) | len_ratio=1.0363
   S10 -> S1   | rot=  -0.0354 deg | mid_t=(   -3.50,    -0.50) | len_ratio=0.9973
   S11 -> S2   | rot=   0.1079 deg | mid_t=(   -3.00,    -2.00) | len_ratio=0.9970
   S12 -> S3   | rot=  -0.0334 deg | mid_t=(   -3.50,    -1.50) | len_ratio=1.0024
   S13 -> S4   | rot=  -0.2427 deg | mid_t=(   -4.50,    -3.00) | len_ratio=0.9394
   S14 -> S5   | rot=  -0.0560 deg | mid_t=(   -3.00,    -4.50) | len_ratio=1.0015
   S15 -> S6   | rot=  -0.0959 deg | mid_t=(   -1.50,    -5.00) | len_ratio=0.9961
   S16 -> S7   | rot=  -0.2362 deg | mid_t=(   -2.00,    -0.50) | len_ratio=1.0061
   S17 -> S8   | rot=   0.0025 deg | mid_t=(   -1.00,    -1.50) | len_ratio=0.9916
    S2 -> S11  | rot=  -0.1370 deg | mid_t=(   -2.00,    -3.00) | len_ratio=0.9433
    S3 -> S12  | rot=   0.2016 deg | mid_t=(   -3.00,    -3.00) | len_ratio=1.0064
    S4 -> S13  | rot=   0.1148 deg | mid_t=(    0.50,     7.50) | len_ratio=1.0178
    S5 -> S14  | rot=   0.2115 deg | mid_t=(   -2.00,    -0.50) | len_ratio=1.0069
    S6 -> S15  | rot=  -0.1208 deg | mid_t=(   -2.50,    -2.00) | len_ratio=0.9917
    S7 -> S16  | rot=   0.0484 deg | mid_t=(   -3.00,    -3.50) | len_ratio=1.0016
    S8 -> S17  | rot=  -0.0566 deg | mid_t=(   -1.50,    -1.50) | len_ratio=1.0033
    S9 -> S0   | rot=   0.1227 deg | mid_t=(   -3.50,    -0.50) | len_ratio=1.0096