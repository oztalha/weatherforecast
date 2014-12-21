"""
by Talha Oz
"""
# -*- coding: utf-8 -*-

import collections
legend = collections.OrderedDict({
's1':"I can't tell",
's2':"Negative",
's3':"Neutral / author is just sharing information",
's4':"Positive",
's5':"Tweet not related to weather condition"  ,
'w1':"current (same day) weather",
'w2':"future (forecast)",
'w3':"I can't tell",
'w4':"past weather",
'k1':"clouds",
'k2':"cold",
'k3':"dry",
'k4':"hot",
'k5':"humid",
'k6':"hurricane",
'k7':"I can't tell",
'k8':"ice",
'k9':"other",
'k10':"rain",
'k11':"snow",
'k12':"storms",
'k13':"sun",
'k14':"tornado",
'k15':"wind"
})
legend = collections.OrderedDict(sorted(legend.items(), key=lambda t: t[0]))