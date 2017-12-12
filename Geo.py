from bokeh.io import output_file, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)

map_options = GMapOptions(lat=37.319470, lng=-121.950812, map_type="roadmap", zoom=12)
#cal = 37.733140, lng=-119.187458

plot = GMapPlot(
    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options
)
plot.title.text = "San Francisco"

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
plot.api_key = "AIzaSyDMQV93DAXK1E56fQF2PolO_42Z_FWX4to"

source = ColumnDataSource(
    data=dict(
        #lat=[40, 40.20, 40.29],
        lat=[29.74628,40.73363,40.839599,40.741122,39.985959,38.93904,45.01051,43.6879119,-37.68624885,29.72425,37.75506412,41.668791,41.668791,36.6786,37.9799,33.7104,37.571,33.786,37.0418,37.7776,33.1951,34.2072,33.949,37.6562,37.3873,40.4168,37.979,37.79337852,38.60888,26.11672,32.90035,40.75987,33.12704,43.6431913,43.6431913,37.34098033,38.61124,32.8611,32.73231405,40.7883,32.56812,33.98,39.254703,33.92,32.7153,39.59471,34.09,41.769513,41.769513,33.84,33.74811,38.496395,28.41635,28.41635,36.19601,32.56812,40.80244,33.02155,43.6671859,43.6671859,43.6671859,43.6224734,43.7440502,40.700542,40.700542,40.694599,33.46083,29.40244,37.79600661,37.79149,45.516801,41.757432,41.757432,53.355957,31.80239,45.497167,34.08383,38.66648,37.7749295,45.553016,40.6916,40.6916,52.071453,34.01979,45.49738,37.78536343,37.78536343,34.5832197,27.96507503,37.80194444,52.45761,51.5365,40.737598,40.737598,10.3236,51.75552,40.79575,40.79575,55.852472,52.36422,47.35671916,51.87142,51.29079,52.0373,52.41962,40.6916,40.6916,53.17113,36.0224,52.08064,52.30981,52.04259,51.67622,52.22883,52.26685,45.461155,47.35671916,52.57328,52.00666,52.54774,31.80239,52.960308,51.85206,51.91363,52.12288,52.08471,46.76647042,52.65863,52.46472,52.17316,51.70175,52.06848,37.78402125,51.55714,52.01394,51.63495,52.49347,51.8347,-28.4832,51.9252,52.22883,52.61531,52.39475,52.36451,46.76647042,45.518033,45.483292,33.673,40.706323,40.706323,40.847641,40.847641,52.62663,52.4125,52.48775,52.09914,-28.4832,33.96169,45.573199,45.573199,40.7891,40.7891,51.85206,50.89829,51.87817,50.867733,45.585347,52.11982,51.87458,52.14205,51.92359,52.02189,53.17309,45.509618,40.79575,40.79575,34.0837402,52.47213,52.09396,45.587697,39.26418,52.16856,52.51293,52.11291,52.18121,52.10302,39.95472,37.775,53.427628,52.48517,52.47186,52.37213,37.5482697,52.16864,38.93952353,41.749759,41.749759,40.700542,40.700542,51.46551,52.67255,39.92944,40.766209,41.749759,40.701527,40.701527,36.19591,53.46709483,41.705819,41.705819,42.994014,42.994014,41.705819,41.747811,41.747811,33.65749,52.29193,51.54987,40.5839,40.5839,45.555284,40.701527,40.701527,40.700542,40.700542,41.757432,41.757432,40.875351,40.875351,40.766307,40.766307,-28.3098,-28.3098,-33.63508,52.32295,40.681254,40.681254,32.2534,52.21539,39.840477,39.840477,32.82935,52.17316,41.769513,41.769513,40.694999,40.694999,37.332,35.12224,37.775,14.55352833,45.532973,22.53694639,53.709976,35.08479,40.639703,41.517665,40.24821,40.639703,40.578629,40.079846,41.517665,40.637908,40.637908,40.637908,40.637908,40.24821,40.637908,40.637908,40.079846,35.08473,41.542589,41.542589,40.67705,39.873949,39.660062,39.873949,39.660062,42.31954181,39.949439,39.949439,40.66631,40.71617,33.49626,27.96792845,42.05551,33.48412,41.421669,41.421669,43.13085,40.6769,34.075059,52.05411,40.6769,42.96074,40.651587,40.651587,40.783199,40.783199,45.533538,34.01068295,51.925926,42.96074,38.8876,34.0470468,39.70775,40.3977,51.84608,39.26418,41.66643,51.50711486,40.845298,40.845298,40.6769,43.6504548,41.81381,40.67775,40.6539,40.6539,34.89263,29.06357,43.688171,38.59662,42.1945,40.377475,40.377475,39.77949,43.27361,39.24666,35.83716,-37.1,43.688171,35.24342,35.97628,41.167576,41.167576,40.825901,40.825901,41.81381,40.687099,40.687099,40.783199,40.657592,40.763405,41.542589,41.142791,41.025874,41.653458,41.602907,25.92543,35.5834,40.712654,40.736999,40.736999,40.949027,40.949027,39.91251,47.44754808,43.16461,40.85532,43.074501,43.074501,39.72033,26.23142,51.20786,43.11019,32.68369,40.39272,41.538887,41.538887,40.737598,40.737598,41.54605,41.173571,41.173571,25.86659,38.70564,41.319999,41.319999,40.6068,40.6068,39.84699,41.538887,40.694999,40.694999,41.231892,41.173571,14.5810124,41.305301,41.305301,42.58391,32.8377,40.854645,40.854148,41.0335,41.0335,39.28104,52.45762,41.212799,40.727518,41.212799,40.727518,41.1009,38.8876,41.0358,41.0358,42.52284,44.64568912,39.28104,40.739058,40.741699,40.677799,40.677799,40.739058,33.71475,40.85532,41.096783,41.096783,40.758399,40.733538,40.758399,40.749939,40.749939,40.733538,33.71163,40.37923,43.11642,32.96817,41.092505,41.092505,40.885762,40.700401,41.092505,40.885762,41.096783,40.700401,35.65401,40.775833,41.08797,40.7714,40.7714,40.775833,40.885762,41.1009,41.0358,41.0358,41.0335,41.0335,13.75887312,33.91294,25.89911,41.08797,40.305185,40.305185,29.59203,41.109397,41.109397,13.75887312,40.751615,40.812099,40.812099,41.390594,40.830799,40.830799,31.77831,41.01143,37.37137,40.827999,40.827999,32.68369,40.26649,40.4419,41.10571,40.6416,40.856201,40.856201,40.6416,39.75187,42.76865,38.94984384,41.54605,40.7263,40.7263,40.854148,41.93398,29.59203,41.109397,41.109397,40.43152,40.741298,40.741298,41.551348,27.93686,41.119899,41.119899,38.70564,41.263999,41.263999,39.03111,41.011656,41.101938,41.109397,41.101938,41.102654,41.102654,41.011656,39.08215,40.840698,40.8591,40.840698,40.8591,40.10513,40.859132,40.859132,40.55453,33.77734,40.727518,40.727518,40.43152,40.05956,41.10571,42.90369,40.786701,41.212799,41.212799,40.786701,43.15705,39.96587,28.16004494,42.90369,41.510822,41.510822,40.738201,40.738201,41.42092,33.94108,45.553016,35.13642,41.94275,40.79131,42.91979,42.9415,41.94275,35.00144,38.86237,43.0645,30.19648,40.666801,40.770455,40.770455,40.609401,40.666801,40.609401,41.231892,39.76012,40.57224,33.46294,38.99583033,41.390594,41.390594,40.824001,40.824001,39.0422443,41.167576,41.167576,39.9327,33.03821,40.679904,40.679904,40.679904,35.04299,34.89013,41.42092,32.41256,41.01143,41.21259,40.808184,40.808184,40.808184,41.167576,39.31804,40.672147,39.31804,40.63457,41.159198,40.762796,41.119899,40.672147,39.390753,39.679562,40.63457,41.070098,40.4286,35.0426,26.26039,33.58482,41.305301,40.851501,40.42922,39.10431,37.8043637,47.6168,43.18811,43.07997,40.4893,35.78041,43.15644,41.80056,41.263999,40.786701,40.9057,41.70289,36.1358,1.30310555,41.94275,41.168386,41.168386,39.0836795,39.06192477,33.98751,40.56793,40.70100019,41.3861,35.25292,35.88437,41.80056,35.91149,30.36615,40.858536,30.36615,40.741699,40.677734,35.79227,51.74972626,35.1146,41.80056,41.168386,41.168386,25.9962,35.76657,39.85277,40.697624,40.726398,40.57125,40.233088,30.688198,41.228539,41.228539,41.108566,41.101004,41.108566,41.101004,40.675498,43.166599,40.675498,35.76657,40.0409,38.87962,40.274342,30.00656,41.37045,35.30882,41.80056,40.762599,41.468843,40.188232,39.826352,26.26039,39.97347,39.14754,35.79822,30.22079,40.10513,43.08405,40.893857,36.20641,39.873123,30.36615,40.07566,30.41939,30.00656,30.19648,38.2976,35.75302,32.86881,41.1455,40.675498,40.98613,40.675498,39.880638,40.706638,43.7632269,40.77166,30.00656,42.31649,41.53924,35.76657,40.282848,40.885762,40.568218,34.63035,40.69682,39.98562,40.5778,39.97347,43.033401,40.91875,42.46,40.899573,42.1496,40.82549,40.739627,30.44866,35.0323,35.86215,40.575395,40.746094,35.0671,43.1792,42.78309,35.47812,29.59203,30.36615,40.808814,40.828899,40.677734,39.88874,42.94285,35.04299,42.93308,41.108566,41.108566,34.80565,33.74557,41.42061,30.22079,-34.8833,43.1647,35.7725,29.73417,40.891899,42.7711,38.94814,30.30931,38.93879,40.839155,35.53698,41.092505,41.092505,33.56706,35.84651,41.10558,39.21912,39.05083,40.29867,14.64411873,41.80858,28.19518,35.76753,40.475963,40.697624,40.739058,29.93819,33.74557,43.14927,39.7585,35.69304,42.99949,27.794876,27.8791,29.72274,39.860082,40.718341,39.27094,41.092505,41.537066,41.092505,41.537066,41.76438,41.76438,44.85877,32.72429,42.33392,40.186214,40.995509,38.94547,39.75187,41.10558,56.9547,42.91979,42.73718,41.80858,35.02872,37.3382082,47.6168,35.76753,35.02872,29.51162,41.55976,41.09567,41.55976,41.09567,34.94787,39.80228,29.73417,42.78382,43.749974,42.91285,29.72274,41.3861,41.1455,40.04827,38.69278,38.8802,41.55976,41.55976,41.58675,29.59203,30.4799,39.873463,40.700544,40.454828,40.610099,40.597802,44.88052,42.98295,34.89753,35.13642,39.84037,41.92585,32.97901,41.590483,41.590483,40.758399,33.23071,43.7271425,43.775891,41.088001,40.667098,35.12452,41.6255,44.88052,32.420334,43.6636927,39.27094,41.549284,41.549284,28.56869,38.63649,41.159198,42.743359,31.67465,29.57751,45.521703,56.9547,41.712037,41.712037,43.15739,38.52518,41.86838,44.85877,35.84651,41.28071,30.02868,34.02025,42.69981,41.42061,35.75366,42.91325,39.09613,35.76657,39.77378258,40.897716,41.251499,35.44491,34.64043,41.712037,39.75231,26.00764,41.86838,30.15768,34.798,42.9999,40.23441,43.07277,41.21294,42.85458,25.96388,35.79841,41.43758,30.36615,38.43767,39.880638,42.52249,40.89319,39.77401,41.4942,40.188232,40.282848,39.942612,30.30931,39.9505,40.58226,39.85277,41.47749,30.5753,38.63649,40.69895,36.06924,39.873123,40.274342,40.274342,39.98631,40.043757,40.043757,39.810711,40.738786,40.738786,43.1428,38.7644,42.98207,29.41785,41.13588,32.98188,45.51159,43.1193,41.81925,44.86269,41.712037,29.62558,35.82858,47.66961,43.1193,35.4588,29.99148,41.608643,41.608643,52.06168,41.319526,41.319526,33.49626,39.77124,47.66961,41.33305,29.55588,40.840099,39.931529,39.931529,40.475963,40.840099,40.475963,40.828498,41.071104,41.319526,42.86349,42.86349,37.81708403,33.8477,30.43909,39.10307,32.7847,41.565944,41.565944,37.775,51.38251,52.39119,52.630043,52.39135,51.68563,51.92035,36.16833,37.78715193,32.7963826,36.13909,47.53219,39.72233,40.04232,40.682899,27.79194,40.846499,45.508435,29.99709,34.03121,40.62589,54.954918,37.34255,30.22325,44.78889,47.65396981,34.00252341,40.748066,42.932594,42.38607,41.554165,41.554165,42.93247,51.87635,52.33557,40.677734,33.498002,34.00988469,30.22325,26.04804,32.22888,32.51419162,30.22325,27.98142,32.64743,41.104708,41.104708,40.607196
],
        #lon=[-7.70, -7.74, -7.78],
        lon=[ -95.26469, -74.12252, -73.929298, -73.956437, -75.05104, -77.13347, -93.08976, -79.3029828, 144.78730113, -95.33066, -122.4404171, -72.82987, -72.82987, -121.6131, -121.8022, -117.9513, -121.9858, -117.9318, -121.4684, -122.2181, -117.3775, -119.1698, -118.202, -122.4257, -122.0158, -120.6553, -122.3365, -122.40706816, -121.2705, -80.32218, -96.71768, -73.95763, -117.1048, -79.3995549, -79.3995549, -121.90955586, -121.4268, -97.29, -117.25644318, -73.9287, -117.0533, -118.09, -121.3999, -118.35, -117.156, -104.8847, -118.37, -72.667793, -72.667793, -118.34, -117.8433, -121.446463, -81.47408, -81.47408, -115.242, -117.0533, -73.4378, -96.82957, -79.4316966, -79.4316966, -79.4316966, -79.5406208, -79.4067728, -73.808036, -73.808036, -73.811996, -111.889, -98.62822, -122.39352316, -122.40409, -122.678669, -72.66263, -72.66263, -1.2849058, -106.5386, -122.579396, -118.35066, -90.22885, -122.4194155, -122.578444, -73.999099, -73.999099, -0.24319015, -117.5429, -122.568492, -122.40051552, -122.40051552, -118.1381318, -82.74500367, -122.41888889, 4.65605, 5.10294, -73.850601, -73.850601, 123.922, 5.28614, -73.82572, -73.82572, -4.309626, 6.49084, -122.84673898, 5.72729, 5.64194, 4.6998, 4.69402, -73.999099, -73.999099, 6.33327, -115.2435, 4.98868, 5.56092, 5.55464, 5.59849, 5.2315, 5.53518, -122.68495, -122.84673898, 5.86237, 5.0463, 4.69229, -106.5386, -1.2693123, 5.70442, 5.77135, 5.02452, 4.41456, -122.19510793, 6.19834, 4.76785, 5.63679, 5.35106, 4.38373, -122.43305802, 4.86008, 4.3132, 4.90272, 4.77501, 5.91895, 24.677, 4.56795, 5.2315, 5.00274, 5.3333, 4.61954, -122.19510793, -122.681276, -122.522589, -111.9906, -74.061815, -74.061815, -73.934855, -73.934855, 5.00541, 6.01558, 4.62934, 5.06728, 24.677, -84.51974, -122.630651, -122.630651, -73.788299, -73.788299, 5.70442, 5.75035, 5.20877, -1.2130957, -122.717395, 5.02961, 5.73146, 5.00341, 4.27517, 4.65404, 6.34209, -122.681117, -73.82572, -73.82572, -118.2947083, 4.7792, 5.15989, -122.751894, -76.58936, 4.56434, 4.71946, 4.49555, 5.42077, 5.36255, -82.8385, -122.418, -1.248398, 4.74536, 5.48486, 4.84411, -121.9885719, 5.59137, -77.54539523, -72.719569, -72.719569, -73.808036, -73.808036, 6.05669, -2.0235898, -82.98408, -73.981761, -72.719569, -74.138762, -74.138762, -115.1385, -2.28200545, -72.803069, -72.803069, -78.928569, -78.928569, -72.803069, -72.659964, -72.659964, -112.2325, 5.55019, 5.13902, -73.966903, -73.966903, -122.70268, -74.138762, -74.138762, -73.808036, -73.808036, -72.66263, -72.66263, -73.834923, -73.834923, -73.981948, -73.981948, 153.51423, 153.51423, 151.14812, 4.76402, -74.20747, -74.20747, -110.8236, 4.51449, -75.192934, -75.192934, -97.06107, 5.63679, -72.667793, -72.667793, -73.519599, -73.519599, -121.89, -85.29707, -122.418, 121.05013333, -122.670639, 88.35427165, -1.7447292, -85.32349, -74.139103, -73.978523, -74.727997, -74.139103, -73.888048, -74.74854, -73.978523, -74.144808, -74.144808, -74.144808, -74.144808, -74.727997, -74.144808, -74.144808, -74.74854, -85.32345, -72.989698, -72.989698, -73.9387, -75.10221, -74.199169, -75.10221, -74.199169, -82.92750763, -74.115669, -74.115669, -73.53732, -73.82648, -86.9103, -82.72438971, -70.72507, -86.91893, -71.842013, -71.842013, -77.66077, -73.93592, -118.022583, 4.72637, -73.93592, -85.6751, -74.287299, -74.287299, -73.623199, -73.623199, -122.55806, -118.49605338, -0.49433872, -85.6751, -76.9772, -118.28841329, -86.15843, -76.9484, 5.23867, -76.58936, -81.33998, -0.12731805, -73.928001, -73.928001, -73.93592, -79.3718947, -71.44001, -73.93861, -74.008003, -74.008003, -82.16011, -80.94467, -79.3618726, -90.21603, -70.95566, -74.150899, -74.150899, -86.27019, -77.75024, -76.57621, -78.61323, 147.417, -79.3618726, -89.70119, -78.87588, -73.201622, -73.201622, -73.859596, -73.859596, -71.44001, -73.8078, -73.8078, -73.623199, -73.895974, -74.112536, -72.989698, -73.42518, -73.608658, -72.659847, -72.591609, -80.26063, -80.55074, -73.900609, -73.931701, -73.931701, -74.07223, -74.07223, -86.03316, -123.87702145, -78.69173, -73.97035, -76.170799, -76.170799, -86.14908, -80.13634, 6.02339, -77.59989, -97.41415, -79.93371, -72.971559, -72.971559, -73.850601, -73.850601, -73.02694, -73.188898, -73.188898, -80.32287, -77.22347, -73.986297, -73.986297, -74.043998, -74.043998, -86.14173, -72.971559, -73.519599, -73.519599, -73.217822, -73.188898, 121.0594014, -74.0354, -74.0354, -71.15774, -97.09079, -73.96987, -73.9659, -73.942298, -73.942298, -76.55333, 4.83871, -73.809501, -74.021061, -73.809501, -74.021061, -73.577, -76.9772, -73.816398, -73.816398, -71.51252, -63.57884109, -76.55333, -74.067733, -73.783401, -73.803199, -73.803199, -74.067733, -84.2408, -73.97035, -73.604889, -73.604889, -73.854598, -74.127323, -73.854598, -73.715021, -73.715021, -74.127323, -84.21738, -77.00465, -77.55735, -80.04534, -73.447605, -73.447605, -74.132285, -73.815299, -73.447605, -74.132285, -73.604889, -73.815299, -78.70584, -74.044768, -73.45699, -73.875999, -73.875999, -74.044768, -74.132285, -73.577, -73.816398, -73.816398, -73.942298, -73.942298, 100.49735069, -84.28614, -80.22743, -73.45699, -74.493594, -74.493594, -95.38622, -73.404274, -73.404274, 100.49735069, -74.187646, -73.130401, -73.130401, -73.598327, -72.925102, -72.925102, -106.46, -74.20255, -121.91142, -73.843803, -73.843803, -97.41415, -82.92759, -80.00883, -73.42577, -73.2633, -73.872299, -73.872299, -73.2633, -86.15417, -86.08604, -77.51737122, -73.02694, -73.2864, -73.2864, -73.9659, -71.3198, -95.38622, -73.404274, -73.404274, -79.9589, -73.214202, -73.214202, -73.035869, -82.43542, -74.0055, -74.0055, -77.22347, -73.684097, -73.684097, -94.50032, -74.20299, -73.992742, -73.404274, -73.992742, -74.028205, -74.028205, -74.20299, -94.49044, -73.615097, -73.461898, -73.615097, -73.461898, -76.33219, -74.011965, -74.011965, -75.4321, -84.39068, -74.021061, -74.021061, -79.9589, -82.95302, -73.42577, -78.84598, -73.823898, -73.809501, -73.809501, -73.823898, -77.54896, -83.00984, -82.7568313, -78.84598, -74.207206, -74.207206, -73.801597, -73.801597, -81.69567, -84.50432, -122.578444, -80.97783, -71.47563, -73.82276, -78.84351, -85.66736, -71.47563, -85.21095, -77.34446, -85.6093, -81.73932, -73.7658, -74.062725, -74.062725, -74.150299, -73.7658, -74.150299, -73.217822, -86.14409, -75.53995, -112.2032, -77.4147142, -73.598327, -73.598327, -73.835998, -73.835998, -77.52321824, -73.201622, -73.201622, -82.8307, -80.14843, -73.905555, -73.905555, -73.905555, -85.31423, -82.40588, -81.69567, -86.351032, -74.20255, -95.99077, -73.907841, -73.907841, -73.907841, -73.201622, -74.588031, -73.843172, -74.588031, -74.145773, -73.774497, -74.009699, -74.0055, -73.843172, -75.010325, -74.271705, -74.145773, -73.803497, -79.9379, -85.25781, -80.13341, -84.51354, -74.0354, -73.952102, -79.99935, -77.64819, -122.2711137, -122.3444, -77.54414, -86.21631, -79.5985, -78.74844, -77.67892, -71.40308, -73.684097, -73.823898, -73.878097, -93.57604, -86.72501, 103.83618564, -71.47563, -73.199484, -73.199484, -77.51656083, -77.79859556, -84.08425, -75.51882, -73.84211381, -73.600097, -89.68277, -78.74076, -71.40308, -78.86547, -81.5519, -73.976938, -81.5519, -73.727897, -73.938598, -78.74549, -0.33914642, -85.26103, -71.40308, -73.199484, -73.199484, -80.16616, -78.73536, -86.27639, -73.853015, -73.838203, -74.328735, -74.700478, -88.010592, -73.244338, -73.244338, -73.406235, -73.57661, -73.406235, -73.57661, -73.872123, -73.714103, -73.872123, -78.73536, -86.28528, -76.99873, -74.706985, -90.01901, -81.5144, -85.14297, -71.40308, -73.902198, -73.821087, -74.746536, -75.104415, -80.13341, -83.01014, -84.53903, -78.48745, -81.57137, -76.33219, -77.60797, -74.229256, -86.77627, -74.921915, -81.5519, -83.13579, -91.12096, -90.01901, -81.73932, -85.54836, -78.64899, -80.02708, -74.022598, -73.872123, -74.166679, -73.872123, -75.01013, -74.252992, -79.3119676, -73.87584, -90.01901, -71.23643, -81.63151, -78.73536, -74.696729, -74.132285, -74.330292, -82.75629, -73.53712, -82.93927, -75.55197, -83.01014, -78.696899, -74.07244, -78.7006, -74.236067, -79.598701, -74.3211, -74.369338, -97.79095, -85.31869, -78.54887, -74.568412, -74.260005, -82.14431, -77.6806, -86.09725, -97.50964, -95.38622, -81.5519, -74.446065, -73.836402, -73.938598, -75.10505, -86.21249, -85.31423, -78.78357, -73.406235, -73.406235, -82.23047, -84.34916, -81.81538, -81.57137, -56.1667, -77.68278, -78.59995, -95.14639, -73.815902, -73.8936, -77.31193, -97.71008, -94.5086, -74.338527, -97.42416, -73.447605, -73.447605, -84.5339, -78.52239, -73.42566, -76.88032, -76.93032, -75.90092, 120.98364106, -71.44081, -82.73988, -78.46518, -74.406589, -73.853015, -74.067733, -90.37646, -84.34916, -77.54015, -86.18862, -78.63588, -78.85333, -82.661761, -82.70016, -95.50098, -75.072552, -73.765464, -84.35138, -73.447605, -72.967396, -73.447605, -72.967396, -72.645359, -72.645359, -93.35009, -96.76196, -71.79834, -74.721942, -74.308681, -76.85831, -86.15417, -73.42566, -98.309, -78.84351, -77.83932, -71.44081, -85.30992, -121.8863286, -122.3444, -78.46518, -85.30992, -98.53513, -72.645162, -73.44413, -72.645162, -73.44413, -82.26256, -77.02273, -95.14639, -86.12039, -79.463634, -85.65696, -95.50098, -73.600097, -74.022598, -82.99982, -76.87207, -77.369, -72.645162, -72.645162, -72.89962, -95.38622, -91.16, -75.019306, -73.807772, -74.273751, -74.119697, -74.319587, -93.02456, -78.792, -82.34014, -80.97783, -83.09064, -71.45878, -80.07705, -72.901836, -72.901836, -73.854598, -86.80733, -79.5784603, -79.2790186, -73.9897, -73.780403, -85.2326, -93.5515, -93.02456, -86.360536, -79.3906609, -84.35138, -73.061501, -73.061501, -81.22396, -90.39439, -73.774497, -73.761911, -106.336, -98.32122, -122.623032, -98.309, -72.587931, -72.587931, -77.53874, -90.39858, -71.43072, -93.35009, -78.52239, -81.72144, -95.4293, -118.1877, -73.84451, -81.81538, -78.63549, -85.6776, -94.56742, -78.73536, -86.15440177, -74.115747, -73.808998, -80.61359, -92.44458, -72.587931, -86.22531, -80.33996, -71.43072, -97.7902, -92.41489, -78.85281, -77.12807, -85.67003, -95.94178, -85.58531, -80.16562, -78.48744, -81.49785, -81.5519, -90.91215, -75.01013, -83.75333, -74.12031, -86.1542, -81.6858, -74.746536, -74.696729, -75.113479, -97.71008, -83.01606, -74.62206, -86.27639, -81.53631, -91.17055, -90.39439, -73.50994, -86.68916, -74.921915, -74.706985, -74.706985, -82.91416, -74.835071, -74.835071, -75.061205, -74.076662, -74.076662, -77.63061, -90.4827, -87.91657, -98.64967, -81.6508, -96.71267, -122.6671, -87.98622, -71.38958, -93.42502, -72.587931, -98.23978, -78.69043, -122.1872, -87.98622, -97.49239, -90.05878, -72.901032, -72.901032, 5.08675, -72.893253, -72.893253, -86.9103, -86.1561, -122.1872, -81.83642, -95.38746, -73.825401, -74.952648, -74.952648, -74.406589, -73.825401, -74.406589, -73.843002, -73.483623, -72.893253, -78.87412, -78.87412, -122.47722602, -84.36742, -91.18729, -84.49948, -79.9493, -72.140428, -72.140428, -122.418, 5.47119, 4.88161, -2.0485356, 4.8193, 5.34843, 5.57156, -86.68854, -122.4098897, -117.155124, -86.88055, -122.1976, -86.16821, -82.99701, -73.805702, -82.67135, -73.940118, -122.778268, -90.14283, -118.4334, -74.01874, -1.6710918, -121.9012, -97.74738, -93.40672, -117.3864266, -118.48837748, -74.167155, -78.76647, -83.15899, -72.843063, -72.843063, -78.7662, 4.96677, 5.64163, -73.938598, -86.903, -118.49525676, -97.74738, -80.16246, -110.8755, -117.00730996, -97.74738, -82.4536, -96.7813, -73.428283, -73.428283, -74.002616,],
    )
)

circle = Circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, line_color=None)
plot.add_glyph(source, circle)

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
output_file("gmap_plot.html")
show(plot)