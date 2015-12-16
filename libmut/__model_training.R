#######################################################################################################################
#model specification for tree based models (RF, GBM, extraTrees)
mdl_def_tree<-paste("T1_V1+T1_V2+T1_V3+T1_V4+T1_V5+T1_V6+T1_V7+T1_V8",
                    "+T1_V9+T1_V10+T1_V11+T1_V12+T1_V13+T1_V14+T1_V15+T1_V16+T1_V17+T2_V1",          
                    "+T2_V2+T2_V3+T2_V4+T2_V5+T2_V6+T2_V7+T2_V8+T2_V9+T2_V10+T2_V11",         
                    "+T2_V12+T2_V13+T2_V14+T2_V15",
                    
                    "+T1_V1_cnt+T1_V2_cnt+T1_V3_cnt+T1_V4_cnt+T1_V5_cnt+T1_V6_cnt+T1_V7_cnt+T1_V8_cnt+T1_V9_cnt+T1_V10_cnt+T1_V11_cnt+T1_V12_cnt+T1_V13_cnt+T1_V14_cnt",     
                    "+T1_V15_cnt+T1_V16_cnt+T1_V17_cnt+T2_V1_cnt+T2_V2_cnt+T2_V3_cnt+T2_V4_cnt+T2_V5_cnt+T2_V6_cnt+T2_V7_cnt",      
                    "+T2_V8_cnt+T2_V9_cnt+T2_V10_cnt+T2_V11_cnt+T2_V12_cnt+T2_V13_cnt+T2_V14_cnt+T2_V15_cnt",
                    
                    "+t1v4_t1v5_cnt+t1v4_t1v6_cnt",  
                    "+t1v4_t1v7_cnt+t1v4_t1v8_cnt+t1v4_t1v9_cnt+t1v4_t1v11_cnt+t1v4_t1v12_cnt+t1v4_t1v15_cnt+t1v4_t1v16_cnt+t1v4_t1v17_cnt+t1v4_t2v3_cnt",  
                    "+t1v4_t2v5_cnt+t1v4_t2v11_cnt+t1v4_t2v12_cnt+t1v4_t2v13_cnt+t1v5_t1v6_cnt+t1v5_t1v7_cnt+t1v5_t1v8_cnt+t1v5_t1v9_cnt+t1v5_t1v11_cnt+t1v5_t1v12_cnt", 
                    "+t1v5_t1v15_cnt+t1v5_t1v16_cnt+t1v5_t1v17_cnt+t1v5_t2v3_cnt+t1v5_t2v5_cnt+t1v5_t2v11_cnt+t1v5_t2v12_cnt+t1v5_t2v13_cnt+t1v6_t1v7_cnt+t1v6_t1v8_cnt",  
                    "+t1v6_t1v9_cnt+t1v6_t1v11_cnt+t1v6_t1v12_cnt+t1v6_t1v15_cnt+t1v6_t1v16_cnt+t1v6_t1v17_cnt+t1v6_t2v3_cnt+t1v6_t2v5_cnt+t1v6_t2v11_cnt+t1v6_t2v12_cnt", 
                    "+t1v6_t2v13_cnt+t1v7_t1v8_cnt+t1v7_t1v9_cnt+t1v7_t1v11_cnt+t1v7_t1v12_cnt+t1v7_t1v15_cnt+t1v7_t1v16_cnt+t1v7_t1v17_cnt+t1v7_t2v3_cnt+t1v7_t2v5_cnt",  
                    "+t1v7_t2v11_cnt+t1v7_t2v12_cnt+t1v7_t2v13_cnt+t1v8_t1v9_cnt+t1v8_t1v11_cnt+t1v8_t1v12_cnt+t1v8_t1v15_cnt+t1v8_t1v16_cnt+t1v8_t1v17_cnt+t1v8_t2v3_cnt",  
                    "+t1v8_t2v5_cnt+t1v8_t2v11_cnt+t1v8_t2v12_cnt+t1v8_t2v13_cnt+t1v9_t1v11_cnt+t1v9_t1v12_cnt+t1v9_t1v15_cnt+t1v9_t1v16_cnt+t1v9_t1v17_cnt+t1v9_t2v3_cnt",  
                    "+t1v9_t2v5_cnt+t1v9_t2v11_cnt+t1v9_t2v12_cnt+t1v9_t2v13_cnt+t1v11_t1v12_cnt+t1v11_t1v15_cnt+t1v11_t1v16_cnt+t1v11_t1v17_cnt+t1v11_t2v3_cnt+t1v11_t2v5_cnt", 
                    "+t1v11_t2v11_cnt+t1v11_t2v12_cnt+t1v11_t2v13_cnt+t1v12_t1v15_cnt+t1v12_t1v16_cnt+t1v12_t1v17_cnt+t1v12_t2v3_cnt+t1v12_t2v5_cnt+t1v12_t2v11_cnt+t1v12_t2v12_cnt",
                    "+t1v12_t2v13_cnt+t1v15_t1v16_cnt+t1v15_t1v17_cnt+t1v15_t2v3_cnt+t1v15_t2v5_cnt+t1v15_t2v11_cnt+t1v15_t2v12_cnt+t1v15_t2v13_cnt+t1v16_t1v17_cnt+t1v16_t2v3_cnt", 
                    "+t1v16_t2v5_cnt+t1v16_t2v11_cnt+t1v16_t2v12_cnt+t1v16_t2v13_cnt+t1v17_t2v3_cnt+t1v17_t2v5_cnt+t1v17_t2v11_cnt+t1v17_t2v12_cnt+t1v17_t2v13_cnt+t2v3_t2v5_cnt",  
                    "+t2v3_t2v11_cnt+t2v3_t2v12_cnt+t2v3_t2v13_cnt+t2v5_t2v11_cnt+t2v5_t2v12_cnt+t2v5_t2v13_cnt+t2v11_t2v12_cnt+t2v11_t2v13_cnt+t2v12_t2v13_cnt",
                    
                    "+exp_t1v4+exp_t1v5+exp_t1v6",       
                    "+exp_t1v7+exp_t1v8+exp_t1v9+exp_t1v11+exp_t1v12+exp_t1v15+exp_t1v16",      
                    "+exp_t1v17+exp_t2v3+exp_t2v5+exp_t2v11+exp_t2v12+exp_t2v13+exp_t1v4_t1v5+exp_t1v4_t1v9+exp_t1v4_t1v11", 
                    "+exp_t1v4_t1v15+exp_t1v4_t1v16+exp_t1v4_t2v5+exp_t1v4_t2v13+exp_t1v5_t1v9+exp_t1v5_t1v11+exp_t1v5_t1v15+exp_t1v5_t1v16+exp_t1v5_t2v5+exp_t1v5_t2v13", 
                    "+exp_t1v9_t1v11+exp_t1v9_t1v15+exp_t1v9_t1v16+exp_t1v9_t2v5+exp_t1v59_t2v13+exp_t1v11_t1v15+exp_t1v11_t1v16+exp_t1v11_t2v5+exp_t1v11_t2v13+exp_t1v15_t1v16",
                    "+exp_t1v15_t2v5+exp_t1v15_t2v13+exp_t1v16_t2v5+exp_t1v16_t2v13+exp_t2v5_t2v13", 
                    
                    "+mean_T1_V4+median_T1_V4+sd_T1_V4+skewness_T1_V4+kurtosis_T1_V4+mean_T1_V5+median_T1_V5+sd_T1_V5",       
                    "+skewness_T1_V5+kurtosis_T1_V5+mean_T1_V6+median_T1_V6+sd_T1_V6+skewness_T1_V6+kurtosis_T1_V6+mean_T1_V7+median_T1_V7",   
                    "+sd_T1_V7+skewness_T1_V7+kurtosis_T1_V7+mean_T1_V8+median_T1_V8+sd_T1_V8+skewness_T1_V8+kurtosis_T1_V8+mean_T1_V9",     
                    "+median_T1_V9+sd_T1_V9+skewness_T1_V9+kurtosis_T1_V9+mean_T1_V11+median_T1_V11+sd_T1_V11+skewness_T1_V11+kurtosis_T1_V11",
                    "+mean_T1_V12+median_T1_V12+sd_T1_V12+skewness_T1_V12+kurtosis_T1_V12+mean_T1_V15+median_T1_V15+sd_T1_V15+skewness_T1_V15",
                    "+kurtosis_T1_V15+mean_T1_V16+median_T1_V16+sd_T1_V16+skewness_T1_V16+kurtosis_T1_V16+mean_T1_V17+median_T1_V17+sd_T1_V17",      
                    "+skewness_T1_V17+kurtosis_T1_V17+mean_T2_V3+median_T2_V3+sd_T2_V3+skewness_T2_V3+kurtosis_T2_V3+mean_T2_V5+median_T2_V5",   
                    "+sd_T2_V5+skewness_T2_V5+kurtosis_T2_V5+mean_T2_V11+median_T2_V11+sd_T2_V11+skewness_T2_V11+kurtosis_T2_V11+mean_T2_V12",    
                    "+median_T2_V12+sd_T2_V12+skewness_T2_V12+kurtosis_T2_V12+mean_T2_V13+median_T2_V13+sd_T2_V13+skewness_T2_V13+kurtosis_T2_V13"
)

#######################################################################################################################
#model specification for tree based models (RF, GBM, extraTrees)
mdl_def_tree_nof<-paste("T1_V1+T1_V2+T1_V3",
                    "+T1_V10+T1_V14+T2_V1",          
                    "+T2_V2+T2_V4+T2_V6+T2_V7+T2_V8+T2_V9+T2_V10",         
                    "+T2_V12+T2_V14+T2_V15",
                    
                    "+T1_V1_cnt+T1_V2_cnt+T1_V3_cnt+T1_V4_cnt+T1_V5_cnt+T1_V6_cnt+T1_V7_cnt+T1_V8_cnt+T1_V9_cnt+T1_V10_cnt+T1_V11_cnt+T1_V12_cnt+T1_V13_cnt+T1_V14_cnt",     
                    "+T1_V15_cnt+T1_V16_cnt+T1_V17_cnt+T2_V1_cnt+T2_V2_cnt+T2_V3_cnt+T2_V4_cnt+T2_V5_cnt+T2_V6_cnt+T2_V7_cnt",      
                    "+T2_V8_cnt+T2_V9_cnt+T2_V10_cnt+T2_V11_cnt+T2_V12_cnt+T2_V13_cnt+T2_V14_cnt+T2_V15_cnt",
                    
                    "+t1v4_t1v5_cnt+t1v4_t1v6_cnt",  
                    "+t1v4_t1v7_cnt+t1v4_t1v8_cnt+t1v4_t1v9_cnt+t1v4_t1v11_cnt+t1v4_t1v12_cnt+t1v4_t1v15_cnt+t1v4_t1v16_cnt+t1v4_t1v17_cnt+t1v4_t2v3_cnt",  
                    "+t1v4_t2v5_cnt+t1v4_t2v11_cnt+t1v4_t2v12_cnt+t1v4_t2v13_cnt+t1v5_t1v6_cnt+t1v5_t1v7_cnt+t1v5_t1v8_cnt+t1v5_t1v9_cnt+t1v5_t1v11_cnt+t1v5_t1v12_cnt", 
                    "+t1v5_t1v15_cnt+t1v5_t1v16_cnt+t1v5_t1v17_cnt+t1v5_t2v3_cnt+t1v5_t2v5_cnt+t1v5_t2v11_cnt+t1v5_t2v12_cnt+t1v5_t2v13_cnt+t1v6_t1v7_cnt+t1v6_t1v8_cnt",  
                    "+t1v6_t1v9_cnt+t1v6_t1v11_cnt+t1v6_t1v12_cnt+t1v6_t1v15_cnt+t1v6_t1v16_cnt+t1v6_t1v17_cnt+t1v6_t2v3_cnt+t1v6_t2v5_cnt+t1v6_t2v11_cnt+t1v6_t2v12_cnt", 
                    "+t1v6_t2v13_cnt+t1v7_t1v8_cnt+t1v7_t1v9_cnt+t1v7_t1v11_cnt+t1v7_t1v12_cnt+t1v7_t1v15_cnt+t1v7_t1v16_cnt+t1v7_t1v17_cnt+t1v7_t2v3_cnt+t1v7_t2v5_cnt",  
                    "+t1v7_t2v11_cnt+t1v7_t2v12_cnt+t1v7_t2v13_cnt+t1v8_t1v9_cnt+t1v8_t1v11_cnt+t1v8_t1v12_cnt+t1v8_t1v15_cnt+t1v8_t1v16_cnt+t1v8_t1v17_cnt+t1v8_t2v3_cnt",  
                    "+t1v8_t2v5_cnt+t1v8_t2v11_cnt+t1v8_t2v12_cnt+t1v8_t2v13_cnt+t1v9_t1v11_cnt+t1v9_t1v12_cnt+t1v9_t1v15_cnt+t1v9_t1v16_cnt+t1v9_t1v17_cnt+t1v9_t2v3_cnt",  
                    "+t1v9_t2v5_cnt+t1v9_t2v11_cnt+t1v9_t2v12_cnt+t1v9_t2v13_cnt+t1v11_t1v12_cnt+t1v11_t1v15_cnt+t1v11_t1v16_cnt+t1v11_t1v17_cnt+t1v11_t2v3_cnt+t1v11_t2v5_cnt", 
                    "+t1v11_t2v11_cnt+t1v11_t2v12_cnt+t1v11_t2v13_cnt+t1v12_t1v15_cnt+t1v12_t1v16_cnt+t1v12_t1v17_cnt+t1v12_t2v3_cnt+t1v12_t2v5_cnt+t1v12_t2v11_cnt+t1v12_t2v12_cnt",
                    "+t1v12_t2v13_cnt+t1v15_t1v16_cnt+t1v15_t1v17_cnt+t1v15_t2v3_cnt+t1v15_t2v5_cnt+t1v15_t2v11_cnt+t1v15_t2v12_cnt+t1v15_t2v13_cnt+t1v16_t1v17_cnt+t1v16_t2v3_cnt", 
                    "+t1v16_t2v5_cnt+t1v16_t2v11_cnt+t1v16_t2v12_cnt+t1v16_t2v13_cnt+t1v17_t2v3_cnt+t1v17_t2v5_cnt+t1v17_t2v11_cnt+t1v17_t2v12_cnt+t1v17_t2v13_cnt+t2v3_t2v5_cnt",  
                    "+t2v3_t2v11_cnt+t2v3_t2v12_cnt+t2v3_t2v13_cnt+t2v5_t2v11_cnt+t2v5_t2v12_cnt+t2v5_t2v13_cnt+t2v11_t2v12_cnt+t2v11_t2v13_cnt+t2v12_t2v13_cnt",
                    
                    "+exp_t1v4+exp_t1v5+exp_t1v6",       
                    "+exp_t1v7+exp_t1v8+exp_t1v9+exp_t1v11+exp_t1v12+exp_t1v15+exp_t1v16",      
                    "+exp_t1v17+exp_t2v3+exp_t2v5+exp_t2v11+exp_t2v12+exp_t2v13+exp_t1v4_t1v5+exp_t1v4_t1v9+exp_t1v4_t1v11", 
                    "+exp_t1v4_t1v15+exp_t1v4_t1v16+exp_t1v4_t2v5+exp_t1v4_t2v13+exp_t1v5_t1v9+exp_t1v5_t1v11+exp_t1v5_t1v15+exp_t1v5_t1v16+exp_t1v5_t2v5+exp_t1v5_t2v13", 
                    "+exp_t1v9_t1v11+exp_t1v9_t1v15+exp_t1v9_t1v16+exp_t1v9_t2v5+exp_t1v59_t2v13+exp_t1v11_t1v15+exp_t1v11_t1v16+exp_t1v11_t2v5+exp_t1v11_t2v13+exp_t1v15_t1v16",
                    "+exp_t1v15_t2v5+exp_t1v15_t2v13+exp_t1v16_t2v5+exp_t1v16_t2v13+exp_t2v5_t2v13", 
                    
                    "+mean_T1_V4+median_T1_V4+sd_T1_V4+skewness_T1_V4+kurtosis_T1_V4+mean_T1_V5+median_T1_V5+sd_T1_V5",       
                    "+skewness_T1_V5+kurtosis_T1_V5+mean_T1_V6+median_T1_V6+sd_T1_V6+skewness_T1_V6+kurtosis_T1_V6+mean_T1_V7+median_T1_V7",   
                    "+sd_T1_V7+skewness_T1_V7+kurtosis_T1_V7+mean_T1_V8+median_T1_V8+sd_T1_V8+skewness_T1_V8+kurtosis_T1_V8+mean_T1_V9",     
                    "+median_T1_V9+sd_T1_V9+skewness_T1_V9+kurtosis_T1_V9+mean_T1_V11+median_T1_V11+sd_T1_V11+skewness_T1_V11+kurtosis_T1_V11",
                    "+mean_T1_V12+median_T1_V12+sd_T1_V12+skewness_T1_V12+kurtosis_T1_V12+mean_T1_V15+median_T1_V15+sd_T1_V15+skewness_T1_V15",
                    "+kurtosis_T1_V15+mean_T1_V16+median_T1_V16+sd_T1_V16+skewness_T1_V16+kurtosis_T1_V16+mean_T1_V17+median_T1_V17+sd_T1_V17",      
                    "+skewness_T1_V17+kurtosis_T1_V17+mean_T2_V3+median_T2_V3+sd_T2_V3+skewness_T2_V3+kurtosis_T2_V3+mean_T2_V5+median_T2_V5",   
                    "+sd_T2_V5+skewness_T2_V5+kurtosis_T2_V5+mean_T2_V11+median_T2_V11+sd_T2_V11+skewness_T2_V11+kurtosis_T2_V11+mean_T2_V12",    
                    "+median_T2_V12+sd_T2_V12+skewness_T2_V12+kurtosis_T2_V12+mean_T2_V13+median_T2_V13+sd_T2_V13+skewness_T2_V13+kurtosis_T2_V13"
)
##################################################################################################
## Original Model Specs
mdl_def_orig<-paste(
                        "T1_V1+T1_V2+T1_V3",
                        "+T1_V10+T1_V14+T2_V1",          
                        "+T2_V2+T2_V4+T2_V6+T2_V7+T2_V8+T2_V9+T2_V10",         
                        "+T2_V12+T2_V14+T2_V15",
                        
                      "+exp_t1v4+exp_t1v5+exp_t1v6",       
                      "+exp_t1v7+exp_t1v8+exp_t1v9+exp_t1v11+exp_t1v12+exp_t1v15+exp_t1v16",      
                      "+exp_t1v17+exp_t2v3+exp_t2v5+exp_t2v11+exp_t2v12+exp_t2v13",
                      "+exp_t1v4_t1v5+exp_t1v4_t1v9+exp_t1v4_t1v11", 

                          "+exp_t1v4_t1v15+exp_t1v4_t1v16+exp_t1v4_t2v5+exp_t1v4_t2v13+exp_t1v5_t1v9+exp_t1v5_t1v11+exp_t1v5_t1v15+exp_t1v5_t1v16+exp_t1v5_t2v5+exp_t1v5_t2v13", 
                          "+exp_t1v9_t1v11+exp_t1v9_t1v15+exp_t1v9_t1v16+exp_t1v9_t2v5+exp_t1v9_t2v13+exp_t1v11_t1v15+exp_t1v11_t1v16+exp_t1v11_t2v5+exp_t1v11_t2v13+exp_t1v15_t1v16",
                          "+exp_t1v15_t2v5+exp_t1v15_t2v13+exp_t1v16_t2v5+exp_t1v16_t2v13+exp_t2v5_t2v13", 

                        "+T1_V1_cnt+T1_V2_cnt+T1_V3_cnt+T1_V4_cnt+T1_V5_cnt+T1_V6_cnt+T1_V7_cnt+T1_V8_cnt+T1_V9_cnt+T1_V10_cnt+T1_V11_cnt+T1_V12_cnt+T1_V13_cnt+T1_V14_cnt",     
                        "+T1_V15_cnt+T1_V16_cnt+T1_V17_cnt+T2_V1_cnt+T2_V2_cnt+T2_V3_cnt+T2_V4_cnt+T2_V5_cnt+T2_V6_cnt+T2_V7_cnt",      
                        "+T2_V8_cnt+T2_V9_cnt+T2_V10_cnt+T2_V11_cnt+T2_V12_cnt+T2_V13_cnt+T2_V14_cnt+T2_V15_cnt",    
                        
#                         "+t1v4_t1v5_cnt+t1v4_t1v6_cnt",  
#                         "+t1v4_t1v7_cnt+t1v4_t1v8_cnt+t1v4_t1v9_cnt+t1v4_t1v11_cnt+t1v4_t1v12_cnt+t1v4_t1v15_cnt+t1v4_t1v16_cnt+t1v4_t1v17_cnt+t1v4_t2v3_cnt",  
#                         "+t1v4_t2v5_cnt+t1v4_t2v11_cnt+t1v4_t2v12_cnt+t1v4_t2v13_cnt+t1v5_t1v6_cnt+t1v5_t1v7_cnt+t1v5_t1v8_cnt+t1v5_t1v9_cnt+t1v5_t1v11_cnt+t1v5_t1v12_cnt", 
#                         "+t1v5_t1v15_cnt+t1v5_t1v16_cnt+t1v5_t1v17_cnt+t1v5_t2v3_cnt+t1v5_t2v5_cnt+t1v5_t2v11_cnt+t1v5_t2v12_cnt+t1v5_t2v13_cnt+t1v6_t1v7_cnt+t1v6_t1v8_cnt",  
#                         "+t1v6_t1v9_cnt+t1v6_t1v11_cnt+t1v6_t1v12_cnt+t1v6_t1v15_cnt+t1v6_t1v16_cnt+t1v6_t1v17_cnt+t1v6_t2v3_cnt+t1v6_t2v5_cnt+t1v6_t2v11_cnt+t1v6_t2v12_cnt", 
#                         "+t1v6_t2v13_cnt+t1v7_t1v8_cnt+t1v7_t1v9_cnt+t1v7_t1v11_cnt+t1v7_t1v12_cnt+t1v7_t1v15_cnt+t1v7_t1v16_cnt+t1v7_t1v17_cnt+t1v7_t2v3_cnt+t1v7_t2v5_cnt",  
#                         "+t1v7_t2v11_cnt+t1v7_t2v12_cnt+t1v7_t2v13_cnt+t1v8_t1v9_cnt+t1v8_t1v11_cnt+t1v8_t1v12_cnt+t1v8_t1v15_cnt+t1v8_t1v16_cnt+t1v8_t1v17_cnt+t1v8_t2v3_cnt",  
#                         "+t1v8_t2v5_cnt+t1v8_t2v11_cnt+t1v8_t2v12_cnt+t1v8_t2v13_cnt+t1v9_t1v11_cnt+t1v9_t1v12_cnt+t1v9_t1v15_cnt+t1v9_t1v16_cnt+t1v9_t1v17_cnt+t1v9_t2v3_cnt",  
#                         "+t1v9_t2v5_cnt+t1v9_t2v11_cnt+t1v9_t2v12_cnt+t1v9_t2v13_cnt+t1v11_t1v12_cnt+t1v11_t1v15_cnt+t1v11_t1v16_cnt+t1v11_t1v17_cnt+t1v11_t2v3_cnt+t1v11_t2v5_cnt", 
#                         "+t1v11_t2v11_cnt+t1v11_t2v12_cnt+t1v11_t2v13_cnt+t1v12_t1v15_cnt+t1v12_t1v16_cnt+t1v12_t1v17_cnt+t1v12_t2v3_cnt+t1v12_t2v5_cnt+t1v12_t2v11_cnt+t1v12_t2v12_cnt",
#                         "+t1v12_t2v13_cnt+t1v15_t1v16_cnt+t1v15_t1v17_cnt+t1v15_t2v3_cnt+t1v15_t2v5_cnt+t1v15_t2v11_cnt+t1v15_t2v12_cnt+t1v15_t2v13_cnt+t1v16_t1v17_cnt+t1v16_t2v3_cnt", 
#                         "+t1v16_t2v5_cnt+t1v16_t2v11_cnt+t1v16_t2v12_cnt+t1v16_t2v13_cnt+t1v17_t2v3_cnt+t1v17_t2v5_cnt+t1v17_t2v11_cnt+t1v17_t2v12_cnt+t1v17_t2v13_cnt+t2v3_t2v5_cnt",  
#                         "+t2v3_t2v11_cnt+t2v3_t2v12_cnt+t2v3_t2v13_cnt+t2v5_t2v11_cnt+t2v5_t2v12_cnt+t2v5_t2v13_cnt+t2v11_t2v12_cnt+t2v11_t2v13_cnt+t2v12_t2v13_cnt",

                        "+mean_T1_V4+median_T1_V4+sd_T1_V4+mean_T1_V5+median_T1_V5+sd_T1_V5",       
                        "+mean_T1_V6+sd_T1_V6+mean_T1_V7+median_T1_V7",   
                        "+sd_T1_V7+mean_T1_V8+median_T1_V8+sd_T1_V8+mean_T1_V9",     
                        "+median_T1_V9+sd_T1_V9+mean_T1_V11+median_T1_V11+sd_T1_V11",
                        "+mean_T1_V12+median_T1_V12+sd_T1_V12+mean_T1_V15+median_T1_V15+sd_T1_V15",
                        "+mean_T1_V16+median_T1_V16+sd_T1_V16+mean_T1_V17+sd_T1_V17",      
                        "+mean_T2_V3+sd_T2_V3+mean_T2_V5+median_T2_V5",   
                        "+sd_T2_V5+mean_T2_V11+sd_T2_V11+mean_T2_V12",    
                        "+sd_T2_V12+mean_T2_V13+median_T2_V13+sd_T2_V13"
#                     "+skewness_T1_V4+kurtosis_T1_V4+skewness_T1_V5+kurtosis_T1_V5+skewness_T1_V6+kurtosis_T1_V6+skewness_T1_V7+kurtosis_T1_V7",
#                     "+skewness_T1_V8+kurtosis_T1_V8+skewness_T1_V9+kurtosis_T1_V9+skewness_T1_V11+kurtosis_T1_V11+skewness_T1_V12+kurtosis_T1_V12",
#                     "+skewness_T1_V15+kurtosis_T1_V15+skewness_T1_V16+kurtosis_T1_V16+skewness_T1_V17+kurtosis_T1_V17+skewness_T2_V3+kurtosis_T2_V3",
#                     "+skewness_T2_V5+kurtosis_T2_V5+skewness_T2_V11+kurtosis_T2_V11+skewness_T2_V12+kurtosis_T2_V12+skewness_T2_V13+kurtosis_T2_V13"
)

#######################################################################################################################
#model specification for glmnet
mdl_def_gnet<-paste("y~log(exp_all_but_res)+log(exp_res)+log(exp_rfd)+log(exp_res_rd)+log(exp_rd+exp_mgr)+log(exp_rid)",
                    "+log(exp_mgr_res)+log(exp_rr2)+log(exp_all_but_res)*log(exp_rid)",
                    "+log(exp_rid)+log(exp_rid)*log(exp_rid)+log(exp_res_rr2)+log(exp_res_rr2)*log(exp_res_rr2)",
                    "+log(exp_mgr_rt)+log(exp_mgr_rt)*log(exp_mgr_rt)+log(exp_res_rd)+log(exp_res_rd)*log(exp_res_rd)",
                    "+log(exp_mgr)+log(exp_mgr)*log(exp_mgr)+log(exp_mgr_rfd)+log(exp_mgr_rfd)*log(exp_mgr_rfd)",
                    "+log(exp_rf)+log(exp_rocd)+log(exp_res_rt)+log(exp_res_rid)+log(exp_res_rr2)+log(exp_res_rt)",
                    "+log(res_rd_cnt-0.5)+log(rid_res_cnt)+log(mgr_rt_cnt)",
                    "+log(exp_all_but_res/(1-exp_all_but_res))+log(exp_all_but_res)*log(exp_all_but_res)",
                    "+log(exp_res)*log(exp_res) + log(exp_res)*log(exp_all_but_res) +log(res_rr2_cnt)",
                    "+log(res_rd_cnt-0.5)*log(rid_res_cnt)",
                    "+log(RESOURCE_cnt)++log(RESOURCE_cnt)*+log(RESOURCE_cnt) +log(all_but_res_cnt)",
                    "+log(mgr_res_cnt)+log(mgr_res_cnt)*log(mgr_res_cnt)",
                    "+log(rt_rd_cnt)+log(rt_rd_cnt)*log(rt_rd_cnt)",
                    "+log(mgr_rd_cnt)+log(mgr_rd_cnt)*log(mgr_rd_cnt)",
                    "+log(ROLE_TITLE_cnt)+log(ROLE_TITLE_cnt)*log(ROLE_TITLE_cnt)",
                    "+log(MGR_ID_cnt)+log(MGR_ID_cnt)*log(MGR_ID_cnt)",
                    "+log(ROLE_DEPTNAME_cnt)+log(ROLE_DEPTNAME_cnt)*log(ROLE_DEPTNAME_cnt)",
                    "+log(ROLE_ROLLUP_2_cnt)+log(ROLE_FAMILY_cnt)+log(ROLE_FAMILY_DESC_cnt)+log(ROLE_ROLLUP_1_cnt)",
                    "+log(ROLE_CODE_cnt)+log(rr2_rd_cnt)",
                    "+log(res_mgr_rr2_cnt)+log(rf_rt_cnt)+log(res_rfd_cnt)+log(rf_rt_cnt)",
                    "+log(exp_rd_rr2)",
                    "+rid_ev1+rid_ev2+rid_ev3+res_ev1+res_ev2+res_ev3+rid_ev4+rid_ev5+res_ev4+res_ev5",
                    "+rid_mgr_ev1+rid_mgr_ev2+rid_mgr_ev3+mgr_rid_ev1+mgr_rid_ev2+mgr_rid_ev3+rid_mgr_ev4+rid_mgr_ev5+mgr_rid_ev4+mgr_rid_ev5",
                    "+mgr_res_ev1+mgr_res_ev2+mgr_res_ev3+res_mgr_ev1+res_mgr_ev2+res_mgr_ev3+mgr_res_ev4+mgr_res_ev5+res_mgr_ev4+res_mgr_ev5"
)
mdl_def_gnet<-paste("y~T1_V1+T1_V2+T1_V3+T1_V4+T1_V5+T1_V6+T1_V7+T1_V8+T1_V9+T1_V10+T1_V11+T1_V12+T1_V13+T1_V14+T1_V15+T1_V16+T1_V17+T2_V1+T2_V2+T2_V3+T2_V4+T2_V5+T2_V6+T2_V7+T2_V8+T2_V9+T2_V10+T2_V11+T2_V12+T2_V13+T2_V14+T2_V15",
                    
                    "+log(exp_t1v4)+log(exp_t1v4)*log(exp_t1v4)+log(exp_t1v5)+log(exp_t1v5)*log(exp_t1v5)+log(exp_t1v6)+log(exp_t1v6)*log(exp_t1v6)",       
                    "+log(exp_t1v7)+log(exp_t1v7)*log(exp_t1v7)+log(exp_t1v8)+log(exp_t1v8)*log(exp_t1v8)+log(exp_t1v9)+log(exp_t1v9)*log(exp_t1v9)+log(exp_t1v11)+log(exp_t1v11)*log(exp_t1v11)+log(exp_t1v12)+log(exp_t1v12)*log(exp_t1v12)",      
                    "+log(exp_t1v15)+log(exp_t1v16)+log(exp_t1v17)+log(exp_t2v3)+log(exp_t2v5)+log(exp_t2v11)+log(exp_t2v12)+log(exp_t2v13)+log(exp_t1v11_t1v16)",
                    "+log(exp_t1v4_t1v5)+log(exp_t1v4_t1v5)*log(exp_t1v4_t1v5)+log(exp_t1v4_t1v11)+log(exp_t1v4_t1v11)*log(exp_t1v4_t1v11)+log(exp_t1v4_t1v15)+log(exp_t1v4_t1v16)+log(exp_t1v5_t1v11)+log(exp_t1v5_t1v15)+log(exp_t1v5_t1v16)", 
                    "+log(exp_t1v11_t1v15)+log(exp_t1v11_t1v15)*log(exp_t1v11_t1v15)+log(exp_t1v15_t1v16)+log(exp_t1v15_t1v16)*log(exp_t1v15_t1v16)",
                    
                    "+log(T1_V1)+log(T1_V2)+log(T1_V3)",
                    "+log(T1_V10)+log(T1_V13)+log(T2_V1)",          
                    "+log(T2_V2)+log(T2_V4)+log(T2_V6)+log(T2_V7)+log(T2_V8)+log(T2_V9)+log(T2_V10)",         
                    "+log(T2_V14)+log(T2_V15)",
                    
                    "+log(T1_V1_cnt)+log(T1_V2_cnt)+log(T1_V3_cnt)+log(T1_V4_cnt)+log(T1_V5_cnt)+log(T1_V6_cnt)+log(T1_V7_cnt)+log(T1_V8_cnt)+log(T1_V9_cnt)+log(T1_V10_cnt)+log(T1_V11_cnt)+log(T1_V12_cnt)+log(T1_V13_cnt)+log(T1_V14_cnt)",     
                    "+log(T1_V15_cnt)+log(T1_V16_cnt)+log(T1_V17_cnt)+log(T2_V1_cnt)+log(T2_V2_cnt)+log(T2_V3_cnt)+log(T2_V4_cnt)+log(T2_V5_cnt)+log(T2_V6_cnt)+log(T2_V7_cnt)",      
                    "+log(T2_V8_cnt)+log(T2_V9_cnt)+log(T2_V10_cnt)+log(T2_V11_cnt)+log(T2_V12_cnt)+log(T2_V13_cnt)+log(T2_V14_cnt)+log(T2_V15_cnt)",
                    
#                     "+log(t1v4_t1v5_cnt)+log(t1v4_t1v6_cnt)",  
#                     "+log(t1v4_t1v7_cnt)+log(t1v4_t1v8_cnt)+log(t1v4_t1v9_cnt)+log(t1v4_t1v11_cnt)+log(t1v4_t1v12_cnt)+log(t1v4_t1v13_cnt)+log(t1v4_t1v15_cnt)+log(t1v4_t1v16_cnt)+log(t1v4_t1v17_cnt)+log(t1v4_t2v3_cnt)",  
#                     "+log(t1v4_t2v5_cnt)+log(t1v4_t2v11_cnt)+log(t1v4_t2v12_cnt)+log(t1v4_t2v13_cnt)+log(t1v5_t1v6_cnt)+log(t1v5_t1v7_cnt)+log(t1v5_t1v8_cnt)+log(t1v5_t1v9_cnt)+log(t1v5_t1v11_cnt)+log(t1v5_t1v12_cnt)", 
#                     "+log(t1v5_t1v15_cnt)+log(t1v5_t1v16_cnt)+log(t1v5_t1v17_cnt)+log(t1v5_t2v3_cnt)+log(t1v5_t2v5_cnt)+log(t1v5_t2v11_cnt)+log(t1v5_t2v12_cnt)+log(t1v5_t2v13_cnt)+log(t1v6_t1v7_cnt)+log(t1v6_t1v8_cnt)",  
#                     "+log(t1v6_t1v9_cnt)+log(t1v6_t1v11_cnt)+log(t1v6_t1v12_cnt)+log(t1v6_t1v15_cnt)+log(t1v6_t1v16_cnt)+log(t1v6_t1v17_cnt)+log(t1v6_t2v3_cnt)+log(t1v6_t2v5_cnt)+log(t1v6_t2v11_cnt)+log(t1v6_t2v12_cnt)", 
#                     "+log(t1v6_t2v13_cnt)+log(t1v7_t1v8_cnt)+log(t1v7_t1v9_cnt)+log(t1v7_t1v11_cnt)+log(t1v7_t1v12_cnt)+log(t1v7_t1v15_cnt)+log(t1v7_t1v16_cnt)+log(t1v7_t1v17_cnt)+log(t1v7_t2v3_cnt)+log(t1v7_t2v5_cnt)",  
#                     "+log(t1v7_t2v11_cnt)+log(t1v7_t2v12_cnt)+log(t1v7_t2v13_cnt)+log(t1v8_t1v9_cnt)+log(t1v8_t1v11_cnt)+log(t1v8_t1v12_cnt)+log(t1v8_t1v15_cnt)+log(t1v8_t1v16_cnt)+log(t1v8_t1v17_cnt)+log(t1v8_t2v3_cnt)",  
#                     "+log(t1v8_t2v5_cnt)+log(t1v8_t2v11_cnt)+log(t1v8_t2v12_cnt)+log(t1v8_t2v13_cnt)+log(t1v9_t1v11_cnt)+log(t1v9_t1v12_cnt)+log(t1v9_t1v15_cnt)+log(t1v9_t1v16_cnt)+log(t1v9_t1v17_cnt)+log(t1v9_t2v3_cnt)",  
#                     "+log(t1v9_t2v5_cnt)+log(t1v9_t2v11_cnt)+log(t1v9_t2v12_cnt)+log(t1v9_t2v13_cnt)+log(t1v11_t1v12_cnt)+log(t1v11_t1v15_cnt)+log(t1v11_t1v16_cnt)+log(t1v11_t1v17_cnt)+log(t1v11_t2v3_cnt)+log(t1v11_t2v5_cnt)", 
#                     "+log(t1v11_t2v11_cnt)+log(t1v11_t2v12_cnt)+log(t1v11_t2v13_cnt)+log(t1v12_t1v15_cnt)+log(t1v12_t1v16_cnt)+log(t1v12_t1v17_cnt)+log(t1v12_t2v3_cnt)+log(t1v12_t2v5_cnt)+log(t1v12_t2v11_cnt)+log(t1v12_t2v12_cnt)",
#                     "+log(t1v12_t2v13_cnt)+log(t1v15_t1v16_cnt)+log(t1v15_t1v17_cnt)+log(t1v15_t2v3_cnt)+log(t1v15_t2v5_cnt)+log(t1v15_t2v11_cnt)+log(t1v15_t2v12_cnt)+log(t1v15_t2v13_cnt)+log(t1v16_t1v17_cnt)+log(t1v16_t2v3_cnt)", 
#                     "+log(t1v16_t2v5_cnt)+log(t1v16_t2v11_cnt)+log(t1v16_t2v12_cnt)+log(t1v16_t2v13_cnt)+log(t1v17_t2v3_cnt)+log(t1v17_t2v5_cnt)+log(t1v17_t2v11_cnt)+log(t1v17_t2v12_cnt)+log(t1v17_t2v13_cnt)+log(t2v3_t2v5_cnt)",  
#                     "+log(t2v3_t2v11_cnt)+log(t2v3_t2v12_cnt)+log(t2v3_t2v13_cnt)+log(t2v5_t2v11_cnt)+log(t2v5_t2v12_cnt)+log(t2v5_t2v13_cnt)+log(t2v11_t2v12_cnt)+log(t2v11_t2v13_cnt)+log(t2v12_t2v13_cnt)"
#                     
                      "+mean_T1_V4+median_T1_V4+sd_T1_V4+mean_T1_V5+median_T1_V5+sd_T1_V5",       
                      "+mean_T1_V6+sd_T1_V6+mean_T1_V7+median_T1_V7",   
                      "+sd_T1_V7+mean_T1_V8+median_T1_V8+sd_T1_V8+mean_T1_V9",     
                      "+median_T1_V9+sd_T1_V9+mean_T1_V11+median_T1_V11+sd_T1_V11",
                      "+mean_T1_V12+median_T1_V12+sd_T1_V12+mean_T1_V15+median_T1_V15+sd_T1_V15",
                      "+mean_T1_V16+median_T1_V16+sd_T1_V16+mean_T1_V17+sd_T1_V17",      
                      "+mean_T2_V3+sd_T2_V3+mean_T2_V5+median_T2_V5",   
                      "+sd_T2_V5+mean_T2_V11+sd_T2_V11+mean_T2_V12",    
                      "+sd_T2_V12+mean_T2_V13+median_T2_V13+sd_T2_V13"

)


mdl_no_cnt <- "y ~ T1_V1+T1_V2+T1_V3+T1_V4+T1_V5+T1_V6+T1_V7+T1_V8+T1_V9+T1_V10+T1_V11+T1_V12+T1_V13+T1_V14+T1_V15+T1_V16+T1_V17+T2_V1+T2_V2+T2_V3+T2_V4+T2_V5+T2_V6+T2_V7+T2_V8+T2_V9+T2_V10+T2_V11+T2_V12+T2_V13+T2_V14+T2_V15"
#######################################################################################################################
#GBM
t_frac<-with(ts1, sum(split1==0)/length(split1))
g2<-gbm(as.formula(paste("y~", mdl_def_tree)),weights=th1$ws, #var.monotone=mono,
        data=ts1, train.fraction=t_frac, n.trees=1000, interaction.depth=8, n.minobsinnode=20, distribution="bernoulli", 
        shrinkage=0.03, bag.fraction=0.5, keep.data=F, verbose=T)

xgb1Ctrl <- trainControl(method="cv",
                         number=7,
                         allowParallel=TRUE,
                         summaryFunction=NormalizedGini)

xgb1Grid <- expand.grid(max_depth = c(5,6,7,8),
                       nrounds = c(400,600,800,1000),
                       eta = c(.015,.03))

m1<-model.matrix(as.formula(mdl_def_gnet), data=ts1)
filter_t<-ts1$split1==0

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
xgb8 <- train(as.formula(paste("Hazard~", mdl_def_orig)), 
              data=ts1[ts1$split1==0,],
              method="xgbTree",
              metric="Gini",
              trControl=xgb1Ctrl,
              tuneGrid=xgb1Grid,
              objective="reg:linear",
              min_child_weight=8,
              subsample=0.8,
              colsample_bytree=0.7,
              verbose=1)
(xgbTime <- Sys.time() - tme)
stopCluster(cl)

pbPost("note", title="xgb Model", body=paste(xgb2$results[1],xgb2$results[2],xgb2$results[3]))
xgb.importance(feature_names=xgb2$finalModel$xNames, model=xgb2$finalModel)
xgb2Imp <- xgb.importance(feature_names=xgb2$finalModel$xNames, model=xgb2$finalModel)
View(xgb2Imp)


########################################################################################333
# Tune GLMNET
glmnetCtrl <- trainControl(method="cv",
                           number=5,
                           repeats=5,
                           allowParallel=TRUE,
                           selectionFunction="best",
                           summaryFunction=NormalizedGini,
                           verboseIter=TRUE)
glmnetGrid <- expand.grid(alpha=c(0.1,0.2,0.4,0.7), lambda=c(.001,.005,.01,.05))

m1<-model.matrix(as.formula(mdl_def_gnet), data=ts1)
filter_t<-ts1$split1==0

set.seed(999)
cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
glmnetTrain <- train(as.formula(paste("Hazard~", mdl_def_orig)),
                     data=ts1[filter_t,],
                     method="glmnet",
                     metric="Gini",
                     trControl=glmnetCtrl,
                     tuneGrid=glmnetGrid)
                     #tuneLength=4)
stopCluster(cl)
Sys.time() - tme
pbPost("note","GLMNET","Done.")
glmnetTrain


rfCtrl <- trainControl(method="cv",
                        number=5,
                        allowParallel=TRUE,
                        selectionFunction="best",
                        summaryFunction=NormalizedGini,
                        verboseIter=TRUE)
rfGrid <- expand.grid(mtry=c(13))

set.seed(1000)
cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
rf2 <- train(as.formula(paste("Hazard~", mdl_def_orig)),
              data=ts1[ts1$split1==0,],
              method="rf",
              metric="Gini",
              trControl=rfCtrl,
              tuneGrid=rfGrid,
              ntree=2000,
              nodesize=20,
              replace=F,
              importance=TRUE,
              do.trace=20
              #maxnodes=500,
              #sampsize=0.5*sum(ts1$split1==0)
              )
stopCluster(cl)
Sys.time() - tme
pbPost("note", title="rf Model", body=paste(rf2$results[1],rf2$results[2],rf2$results[3]))
rf2
save(rf2, file="rf2mod.rda")

#####################################################################################333
## SVM

svmCtrl <- trainControl(method="cv",
                         number=5,
                         allowParallel=TRUE,
                         summaryFunction=NormalizedGini)

svmGrid <- expand.grid(C = c(1,2,4), sigma=c(0.0005))


cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
svm3 <- train(as.formula(paste("Hazard~", mdl_def_orig)), 
              data=ts1[ts1$split1==0,],
              method="svmRadial",
              metric="Gini",
              trControl=svmCtrl,
              tuneGrid=svmGrid,
              preProcess=c("center","scale")
)
(svmTime <- Sys.time() - tme)
stopCluster(cl)

ensCtrl <- trainControl(method="cv",
                        number=5,
                        savePredictions=TRUE,
                        allowParallel=TRUE,
                        index=createMultiFolds(ts1$Hazard[ts1$split==0], k=5, times=5),
                        selectionFunction="best",
                        summaryFunction=NormalizedGini)

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
model_list <- caretList(
  as.formula(paste("Hazard~", mdl_def_orig)),
  data=ts1[ts1$split1==0,],
  trControl=ensCtrl,
  metric="Gini",
  tuneList=list(
#     rf1=caretModelSpec(method="rf", 
#                        tuneGrid=expand.grid(mtry=c(13)), 
#                        nodesize=10, 
#                        ntree=1500),
    rf2=caretModelSpec(method="rf", 
                       tuneGrid=expand.grid(mtry=c(17)), 
                       nodesize=20, 
                       ntree=2000),
    xgb1=caretModelSpec(method="xgbTree", 
                        tuneGrid=expand.grid(max_depth = c(5),
                                             nrounds = c(1000),
                                             eta = c(.015)),
                        preProcess=c("center","scale"),
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=0.8),
    xgb2=caretModelSpec(method="xgbTree", 
                        tuneGrid=expand.grid(max_depth = c(6),
                                             nrounds = c(600),
                                             eta = c(.015)),
                        preProcess=c("center","scale"),
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=0.8),
    xgb3=caretModelSpec(method="xgbTree", 
                        tuneGrid=expand.grid(max_depth = c(3),
                                             nrounds = c(1200),
                                             eta = c(.015)),
                        preProcess=c("center","scale"),
                        min_child_weight=5,
                        subsample=0.7,
                        colsample_bytree=0.7),
    gbm1=caretModelSpec(method="gbm", 
                        preProcess=c("center","scale"),
                        tuneGrid=expand.grid(interaction.depth=c(19),
                                             n.trees=c(3000),
                                             shrinkage=c(.01),
                                             n.minobsinnode=c(20))),
    gbm1=caretModelSpec(method="gbm", 
                        preProcess=c("center","scale"),
                        tuneGrid=expand.grid(interaction.depth=c(13),
                                             n.trees=c(3000),
                                             shrinkage=c(.01),
                                             n.minobsinnode=c(5))),
    svm1=caretModelSpec(method="svmRadial",
                        preProcess=c("center", "scale"),
                        tuneGrid=expand.grid(C = c(4), sigma=c(0.0005))),
    glmnet=caretModelSpec(method="glmnet",
                          preProcess=c("center","scale"),
                          tuneGrid=expand.grid(alpha=c(.2), lambda=c(.05)))
                        
  )
)
stopCluster(cl)
Sys.time() - tme
pbPost("note", "Ensemble", "Finished.")


save(model_list, file="model_list-RF-XGB-GBM-GLMNET-SVM-08-21-2015.rda")

xyplot(resamples(model_list))
modelCor(resamples(model_list))
greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble)

library('caTools')
model_preds <- lapply(model_list, predict, newdata=ts1[split1==2,])
#model_preds <- lapply(model_preds, function(x) x[,'Yes'])
model_preds <- data.frame(model_preds)
ens_preds <- predict(greedy_ensemble, newdata=ts1[ts1$split1==2,])
model_preds$ensemble <- ens_preds

ens_sum_preds <- rowSums(model_preds)

submission = data.frame(Id = ts1$Id[ts1$split1==2], Hazard = ens_preds)
write.csv(submission, "SubmissionEnsemble-rf-xgb-gbm-glmnet-svm-08021015.csv", row.names=FALSE)
