#!/usr/bin/python
# -*- coding: utf-8 -*-

import tools_for_text_analysis as too
#@profile
def main():
    #dataset = too.Parse_data('./cmd.history','two')
    dataset,dataset_show = too.Parse_data('./login_out.csv','three')
    #dataset = too.Merge_data_sliding_window(dataset,50)
    #dataset = too.Merge_data_split_window(dataset)
    vocabset = too.CreateVocabList(dataset)
    #vocabset = too.CalLessFreq(vocabset,dataset)
    #datavec_set = too.BagofWords2Vec(vocabset,dataset)
    datavec_set = too.PolyofWords2Vec(vocabset,dataset)
    #datavec = too.SetofWords2Vec(vocabset,dataset)

    #datavec_mod = too.Modify_counts_with_TFIDF(datavec_set)
    datavec,scaler = too.Data_process(datavec_set)
    #pca_num,ret = too.Gs_PCA(datavec)
    #datavec = too.Model_PCA(datavec,ret['99%'])
    #print pca_num,ret

    #datavec = too.Model_deep_auto_encoder(datavec)
    #datavec = too.Model_deep_auto_encoder_noisy(datavec)
    #best_encoding_dim = too.Gs_auto_encoder_parameter(datavec,20)
    #datavec = too.Model_auto_encoder(datavec,25)
    #datavec = too.Model_variational_autoencoder(datavec)

    #best_epsilon,best_num = too.Gs_DBSCAN_parameter(datavec)
    #clst_labels = too.Model_DBSCAN(datavec,best_epsilon,best_num)
    clst_labels = too.Model_DBSCAN(datavec,0.001,2)
    #datavec = too.Data_inverse_transform(datavec,scaler)

    #show_data = too.Show_data(vocabset,datavec_set,clst_labels,options='two')
    show_data = too.Show_data(vocabset,dataset_show,clst_labels,options='three')
    too.Store_data('./ttt.dat',show_data)
    too.Find_exception('./exception.dat',show_data)
    
if __name__ == "__main__":
    main()
