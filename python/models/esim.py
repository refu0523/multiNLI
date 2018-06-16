import tensorflow as tf
from util import blocks
from functools import reduce
from operator import mul
import pdb


class MyModel(object):
    def __init__(self, seq_length, emb_dim, hidden_dim, embeddings, emb_train):
        ## Define hyperparameters
        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length 

        ## Define the placeholders
        self.premise_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.hypothesis_x = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.premise_pos = tf.placeholder(tf.int32, [None, self.sequence_length, 47], name='premise_pos')
        self.hypothesis_pos = tf.placeholder(tf.int32, [None, self.sequence_length, 47], name='hypothesis_pos')
        self.y = tf.placeholder(tf.int32, [None])
        self.keep_rate_ph = tf.placeholder(tf.float32, [])

        ## Define parameters
        self.E = tf.Variable(embeddings, trainable=emb_train)
        
        self.W_mlp = tf.Variable(tf.random_normal([self.dim * 12, self.dim], stddev=0.1))
        self.b_mlp = tf.Variable(tf.random_normal([self.dim], stddev=0.1))

        self.W_cl = tf.Variable(tf.random_normal([self.dim, 3], stddev=0.1))
        self.b_cl = tf.Variable(tf.random_normal([3], stddev=0.1))

        # ## Define External Knowledge dictionary para.
        # self.exterKnowledge_dic = exterKnowledge_dic
        ## Define R_matrix
        self.R_mat = tf.placeholder(tf.float32, [None, self.sequence_length,self.sequence_length])
        
        ## Function for embedding lookup and dropout at embedding layer
        def emb_drop(x):
            emb = tf.nn.embedding_lookup(self.E, x)
            emb_drop = tf.nn.dropout(emb, self.keep_rate_ph)
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, mask_prem = blocks.length(self.premise_x)
        hyp_seq_lengths, mask_hyp = blocks.length(self.hypothesis_x)


        ### First biLSTM layer ###

        premise_in = tf.concat([emb_drop(self.premise_x), tf.cast(self.premise_pos, tf.float32)], axis=2)
        hypothesis_in =  tf.concat([emb_drop(self.hypothesis_x), tf.cast(self.hypothesis_pos, tf.float32)], axis=2)


        premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
        hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        premise_list = tf.unstack(premise_bi, axis=1)
        hypothesis_list = tf.unstack(hypothesis_bi, axis=1)

        ### self-attention ###
        premise_project = blocks.dense(premise_bi, 600)
        premise_project_list = tf.unstack(premise_project, axis=1)
        premise_self_attn = []
        alphas = []

        for i in range(self.sequence_length):
            scores_i_list = []
            for j in range(self.sequence_length):
                score_ij = tf.reduce_sum(tf.multiply(premise_project_list[i], premise_project_list[j]), 1, keep_dims=True)
                scores_i_list.append(score_ij)
            scores_i = tf.stack(scores_i_list, axis=1)
            alpha_i = blocks.masked_softmax(scores_i, mask_prem)
            p_tilde_i = tf.reduce_sum(tf.multiply(alpha_i, premise_bi), 1)
            premise_self_attn.append(p_tilde_i)


        hypothesis_project = blocks.dense(hypothesis_bi, 600)
        hypothesis_project_list = tf.unstack(hypothesis_project, axis=1)
        hypothesis_self_attn = []
        for i in range(self.sequence_length):
            scores_i_list = []
            for j in range(self.sequence_length):
                score_ij = tf.reduce_sum(tf.multiply(hypothesis_project_list[i], hypothesis_project_list[j]), 1, keep_dims=True)
                scores_i_list.append(score_ij)
            scores_i = tf.stack(scores_i_list, axis=1)
            beta_i = blocks.masked_softmax(scores_i, mask_hyp)
            h_tilde_i = tf.reduce_sum(tf.multiply(beta_i, hypothesis_bi), 1)
            hypothesis_self_attn.append(h_tilde_i)

        premise_self_attns = tf.stack(premise_self_attn, axis=1)
        hypothesis_self_attns = tf.stack(hypothesis_self_attn, axis=1)

        ### Attention ###

        scores_all = []
        premise_attn = []
        alphas = []
        r_alpha = []
        r_all = []

        for i in range(self.sequence_length):
            scores_i_list = []
            r_i_list = []
            for j in range(self.sequence_length):
                #caculate similarity score_ij (e_ij)

                score_ij_ori = tf.reduce_sum(tf.multiply(premise_list[i], hypothesis_list[j]), 1, keep_dims=True)
                ext_r = tf.expand_dims(self.R_mat[:,i,j],axis=1)
                score_ij = score_ij_ori + ext_r
                scores_i_list.append(score_ij)
                r_ij = self.R_mat[:,i,j]
                r_i_list.append(r_ij)
                #pdb.set_trace()             
            scores_i = tf.stack(scores_i_list, axis=1)
            r_i = tf.expand_dims(tf.stack(r_i_list, axis=1), 2)
            #alpha_i: weigth of hypothesis_bi
            alpha_i = blocks.masked_softmax(scores_i, mask_hyp)
            a_tilde_i = tf.reduce_sum(tf.multiply(alpha_i, hypothesis_bi), 1)
            premise_attn.append(a_tilde_i)

            r_alpha_i = tf.reduce_sum(tf.multiply(r_i, alpha_i), 1)
            
            scores_all.append(scores_i)
            alphas.append(alpha_i)
            r_alpha.append(r_alpha_i)
            r_all.append(r_i)

        scores_stack = tf.stack(scores_all, axis=2)
        scores_list = tf.unstack(scores_stack, axis=1) #turn i index to j index

        r_stack = tf.stack(r_all, axis=2)
        r_list = tf.unstack(r_stack, axis=1) #turn i index to j index

        hypothesis_attn = []
        betas = []
        r_beta = []
        for j in range(self.sequence_length):
            scores_j = scores_list[j]
            beta_j = blocks.masked_softmax(scores_j, mask_prem)
            b_tilde_j = tf.reduce_sum(tf.multiply(beta_j, premise_bi), 1)
            hypothesis_attn.append(b_tilde_j)

            r_j = r_list[j]
            r_beta_j = tf.reduce_sum(tf.multiply(r_j, beta_j), 1)
            r_beta.append(r_beta_j)

            betas.append(beta_j)
        # Make r_alpha and r_beta in tensor
        r_alphas = tf.stack(r_alpha, axis=1)
        r_betas = tf.stack(r_beta, axis=1)

        # Make attention-weighted sentence representations into one tensor,
        premise_attns = tf.stack(premise_attn, axis=1)
        hypothesis_attns = tf.stack(hypothesis_attn, axis=1)

        # For making attention plots, 
        self.alpha_s = tf.stack(alphas, axis=2)
        self.beta_s = tf.stack(betas, axis=2) 


        ### Subcomponent Inference ###
        prem_self_diff = tf.subtract(premise_bi, premise_self_attns)
        prem_self_mul = tf.multiply(premise_bi, premise_self_attns)
        hyp_self_diff = tf.subtract(hypothesis_bi, hypothesis_self_attns)
        hyp_self_mul = tf.multiply(hypothesis_bi, hypothesis_self_attns)

        prem_diff = tf.subtract(premise_bi, premise_attns)
        prem_mul = tf.multiply(premise_bi, premise_attns)
        hyp_diff = tf.subtract(hypothesis_bi, hypothesis_attns)
        hyp_mul = tf.multiply(hypothesis_bi, hypothesis_attns)

        ### Factorize Machine ###

        FM_premise_self_attns = tf.expand_dims(blocks.factorize_machine(tf.concat([premise_bi ,premise_self_attns], 2)), 2)
        FM_prem_self_diff = tf.expand_dims(blocks.factorize_machine(prem_self_diff), 2)
        FM_prem_self_mul = tf.expand_dims(blocks.factorize_machine(prem_self_mul), 2)
        
        FM_hypothesis_self_attns = tf.expand_dims(blocks.factorize_machine(tf.concat([hypothesis_bi ,hypothesis_self_attns], 2)), 2)
        FM_hyp_self_diff = tf.expand_dims(blocks.factorize_machine(hyp_self_diff), 2)
        FM_hyp_self_mul = tf.expand_dims(blocks.factorize_machine(hyp_self_mul), 2)


        FM_premise_attns = tf.expand_dims(blocks.factorize_machine(tf.concat([premise_bi ,premise_attns], 2)), 2)
        FM_prem_diff = tf.expand_dims(blocks.factorize_machine(prem_diff), 2)
        FM_prem_mul = tf.expand_dims(blocks.factorize_machine(prem_mul), 2)
        
        FM_hypothesis_attns = tf.expand_dims(blocks.factorize_machine(tf.concat([hypothesis_bi ,hypothesis_attns], 2)), 2)
        FM_hyp_diff = tf.expand_dims(blocks.factorize_machine(hyp_diff), 2)
        FM_hyp_mul = tf.expand_dims(blocks.factorize_machine(hyp_mul), 2)

        m_a = tf.concat([premise_bi, FM_premise_attns, FM_prem_diff, FM_prem_mul,
                         FM_premise_self_attns, FM_prem_self_diff, FM_prem_self_mul, r_alphas], 2)
        m_b = tf.concat([hypothesis_bi, FM_hypothesis_attns, FM_hyp_diff, FM_hyp_mul,
                         FM_hypothesis_self_attns, FM_hyp_self_diff, FM_hyp_self_mul, r_betas], 2)
        
        
        ### Inference Composition ###

        v1_outs, c3 = blocks.biLSTM(m_a, dim=self.dim, seq_len=prem_seq_lengths, name='v1')
        v2_outs, c4 = blocks.biLSTM(m_b, dim=self.dim, seq_len=hyp_seq_lengths, name='v2')

        v1_bi = tf.concat(v1_outs, axis=2)
        v2_bi = tf.concat(v2_outs, axis=2)


        ### Pooling Layer ###

        v_1_sum = tf.reduce_sum(v1_bi, 1)
        v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

        v_2_sum = tf.reduce_sum(v2_bi, 1)
        v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

        v_1_max = tf.reduce_max(v1_bi, 1)
        v_2_max = tf.reduce_max(v2_bi, 1)

        
        alpha_w = blocks.masked_softmax(blocks.dense(r_alphas, 1), mask_prem)
        a_w = tf.reduce_sum(tf.multiply(alpha_w, v1_bi), 1)

        beta_w = blocks.masked_softmax(blocks.dense(r_betas, 1), mask_hyp)
        b_w = tf.reduce_sum(tf.multiply(beta_w, v2_bi), 1)

        v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max, a_w, b_w], 1)
        

        # MLP layer
        h_mlp = tf.nn.tanh(tf.matmul(v, self.W_mlp) + self.b_mlp)

        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl

        # Define the cost function
        self.total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        
       
