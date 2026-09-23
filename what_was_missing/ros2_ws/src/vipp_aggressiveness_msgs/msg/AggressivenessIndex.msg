# ai_score: index under the gate-predicted weights, 0 to 100
# ai_weather: index under the weather regime weights, 0 to 100
# mixture_weights: gate output over (highway, urban, weather), sums to 1
float32 ai_score
float32 ai_weather
float32[] mixture_weights
string vehicle_id
builtin_interfaces/Time stamp
