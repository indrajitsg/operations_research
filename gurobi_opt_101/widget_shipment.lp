\ Model widgets
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  4.5 Baltimore_to_Columbia + 5.09 Baltimore_to_Indianapolis
   + 4.33 Baltimore_to_Lexington + 5.96 Baltimore_to_Nashville
   + 1.96 Baltimore_to_Richmond + 7.3 Baltimore_to_St_Louis
   + 2.43 Cleveland_to_Columbia + 2.37 Cleveland_to_Indianapolis
   + 2.54 Cleveland_to_Lexington + 4.13 Cleveland_to_Nashville
   + 3.2 Cleveland_to_Richmond + 4.88 Cleveland_to_St_Louis
   + 6.42 Little_Rock_to_Columbia + 4.83 Little_Rock_to_Indianapolis
   + 3.39 Little_Rock_to_Lexington + 4.4 Little_Rock_to_Nashville
   + 7.44 Little_Rock_to_Richmond + 2.92 Little_Rock_to_St_Louis
   + 3.33 Birmingham_to_Columbia + 4.33 Birmingham_to_Indianapolis
   + 3.38 Birmingham_to_Lexington + 1.53 Birmingham_to_Nashville
   + 5.95 Birmingham_to_Richmond + 4.01 Birmingham_to_St_Louis
   + 3.02 Charleston_to_Columbia + 2.61 Charleston_to_Indianapolis
   + 1.61 Charleston_to_Lexington + 4.44 Charleston_to_Nashville
   + 2.36 Charleston_to_Richmond + 4.6 Charleston_to_St_Louis
Subject To
 meet_demand[Columbia]: Baltimore_to_Columbia + Cleveland_to_Columbia
   + Little_Rock_to_Columbia + Birmingham_to_Columbia
   + Charleston_to_Columbia >= 89
 meet_demand[Indianapolis]: Baltimore_to_Indianapolis
   + Cleveland_to_Indianapolis + Little_Rock_to_Indianapolis
   + Birmingham_to_Indianapolis + Charleston_to_Indianapolis >= 95
 meet_demand[Lexington]: Baltimore_to_Lexington + Cleveland_to_Lexington
   + Little_Rock_to_Lexington + Birmingham_to_Lexington
   + Charleston_to_Lexington >= 121
 meet_demand[Nashville]: Baltimore_to_Nashville + Cleveland_to_Nashville
   + Little_Rock_to_Nashville + Birmingham_to_Nashville
   + Charleston_to_Nashville >= 101
 meet_demand[Richmond]: Baltimore_to_Richmond + Cleveland_to_Richmond
   + Little_Rock_to_Richmond + Birmingham_to_Richmond
   + Charleston_to_Richmond >= 116
 meet_demand[St_Louis]: Baltimore_to_St_Louis + Cleveland_to_St_Louis
   + Little_Rock_to_St_Louis + Birmingham_to_St_Louis
   + Charleston_to_St_Louis >= 181
 can_produce[Baltimore]: Baltimore_to_Columbia + Baltimore_to_Indianapolis
   + Baltimore_to_Lexington + Baltimore_to_Nashville
   + Baltimore_to_Richmond + Baltimore_to_St_Louis <= 180
 can_produce[Cleveland]: Cleveland_to_Columbia + Cleveland_to_Indianapolis
   + Cleveland_to_Lexington + Cleveland_to_Nashville
   + Cleveland_to_Richmond + Cleveland_to_St_Louis <= 200
 can_produce[Little_Rock]: Little_Rock_to_Columbia
   + Little_Rock_to_Indianapolis + Little_Rock_to_Lexington
   + Little_Rock_to_Nashville + Little_Rock_to_Richmond
   + Little_Rock_to_St_Louis <= 140
 can_produce[Birmingham]: Birmingham_to_Columbia
   + Birmingham_to_Indianapolis + Birmingham_to_Lexington
   + Birmingham_to_Nashville + Birmingham_to_Richmond
   + Birmingham_to_St_Louis <= 80
 can_produce[Charleston]: Charleston_to_Columbia
   + Charleston_to_Indianapolis + Charleston_to_Lexington
   + Charleston_to_Nashville + Charleston_to_Richmond
   + Charleston_to_St_Louis <= 180
 must_produce[Baltimore]: Baltimore_to_Columbia + Baltimore_to_Indianapolis
   + Baltimore_to_Lexington + Baltimore_to_Nashville
   + Baltimore_to_Richmond + Baltimore_to_St_Louis >= 135
 must_produce[Cleveland]: Cleveland_to_Columbia + Cleveland_to_Indianapolis
   + Cleveland_to_Lexington + Cleveland_to_Nashville
   + Cleveland_to_Richmond + Cleveland_to_St_Louis >= 150
 must_produce[Little_Rock]: Little_Rock_to_Columbia
   + Little_Rock_to_Indianapolis + Little_Rock_to_Lexington
   + Little_Rock_to_Nashville + Little_Rock_to_Richmond
   + Little_Rock_to_St_Louis >= 105
 must_produce[Birmingham]: Birmingham_to_Columbia
   + Birmingham_to_Indianapolis + Birmingham_to_Lexington
   + Birmingham_to_Nashville + Birmingham_to_Richmond
   + Birmingham_to_St_Louis >= 60
 must_produce[Charleston]: Charleston_to_Columbia
   + Charleston_to_Indianapolis + Charleston_to_Lexington
   + Charleston_to_Nashville + Charleston_to_Richmond
   + Charleston_to_St_Louis >= 135
Bounds
End
