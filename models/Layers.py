import os 
import random 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable 
import torch.nn.init as init 
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

def set_dropout_prob(p):
    global dropout_p 
    dropout_p=p 

def set_seq_dropout(option):
    global do_seq_dropout
    do_seq_dropout=option 

def seq_dropout(x,p=0,training=False):
    if training==False or p==0:
        return x
    dropout_mask=Variable(1.0/(1-p)*torch.bernoulli((1-p)*(x.data.new(x.size(0),x.size(2)).zero_()+1)),requires_grad=False)
    return dropout_mask.unsqueeze(1).expand_as(x)*x 

def dropout(x,p=0,training=False):
    if do_seq_dropout and len(x,size())==3:
        return seq_dropout(x,p=p,training=training)
    else:
        return F.dropout(x,p=p,training=training)

class CNN(nn.Module):
    def __init__(self,input_size,window_size,output_size):
        super(CNN,self).__init__()
        if window_size% 2!=1:
             raise Exception("window size must be an odd number")
        padding_size=int((window_size-1)/2)
        self._output_size=output_size
        self.cnn=nn.Conv2d(1,output_size,(window_size,input_size),padding=(padding_size,0),bias=False)
        init.xavier_uniform(self.cnn.weight)

    def output_size(self):
        return self._output_size

    def forward(self,x,x_mask):
        x=F.dropout(x,p=dropout_p,training=self.training)
        x_unsqueeze=x.unsqueeze(1)
        x_conv=F.tanh(self.cnn(x_unsqueeze).squeeze(3))
        x_output=torch.transpose(x_conv,1,2)
        return x_output


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling,self).__init__()
        self.MIN=-1e6


    def forward(self,x,x_mask):
        empty_mask=x_mask.eq(0).unsqueeze(2).expand_as(x)
        x_now=x.clone()
        x_now.data.masked_fill_(empty_mask.data,self.MIN)
        x_output=x_now.max(1)[0]
        x_output.data.masked_fill_(x_output.data.eq(self.MIN),0)
        return x_outpout

class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling,self).__init__()

    def forward(self, x,x_mask):
        x_now=x.clone()
        empty_mask = x_mask.eq(0).unsqueeze(2).expand_as(x_now)
        x_now.data.masked_fill_(empty_mask.data, 0)
        x_sum = torch.sum(x_now, 1)
        x_num = torch.sum(x_mask.eq(1).float(), 1).unsqueeze(1).expand_as(x_sum)
        x_num = torch.clamp(x_num, min = 1)

        return x_sum / x_num 



class stackedBRNN(nn.Module):
    def __init__(self,intput_size,hidden_size,num_layers,rnn_type=nn.LSTM,concat_layers=False,bidirectional=True,add_feat=0):
        self.bidir_coef=2 if bidirectional else 1
        self.num_layers=num_layers
        self.concat_layer=concat_layers
        self.hidden_size=hidden_size
        self.rnns=nn.ModuleList()
        for i in range(num_layers):
            in_size=input_size if i==0 else (self.bidir_coef* hidden_size+add_feat if i==1 else self.bidir_coef*hidden_size)
            rnn=rnn_type(in_size,hidden_size,num_layers=1,bidirectional=bidirectional,batch_first=True)
            self.rnns.append(rnn)
    
    def output_size(self):
        if self.concat_layers:
            return self.num_layers * self.bidir_coef * self.hidden_size
        else:
            return self.bidir_coef * self.hidden_size
    
    def forward(self,x,x_mask,return_list=False,x_additional=None):
        hiddens=[x]
        for i in range(self.num_layers):
            rnn_input=hiddens[-1]
            if i==1 and x_additional is not None:
                rnn_input=torch.cat((rnn_input,x_additional),2)

            if dropout_p>0:
                rnn_input=dropout(rnn_input,p=dropout,training=self.training)
            
            rnn_output=self.rnns[i](rnn_input)[0]
            hiddens.append(rnn_output)

        if self.concat_layer:
            output=torch.cat(hiddens[1:],2)
        else:
            output=hiddens[-1]

        if return_list:
            return output,hiddens[1:]
        else:
            return output

class AttentionScore(nn.Module):
    def __init__(self,input_size,hidden_size,correlation_func=1,do_similariy=False):
        super(AttentionScore,self).__init__()
        self.correlation_func=correlation_func
        self.hidden_size=hidden_size

        if correlation_func ==2 or correlation_func==3:
            self.linear=nn.Linear(input_size,hidden_size,bias=False)
            if do_similariy:
                self.diagonal=Parameter(torch.ones(1,1,1)/(hidden_size** 0.5),requires_grad=False)
            else:
                self.diagonal=Parameter(torch.ones(1,1,hidden_size),requires_grad=False)
        
        if correlation_func==4:
            self.linear=nn.Linear(input_size,input_size,bias=False)

        if correlation_func==5:
            self.linear=nn.Linear(input_size,hidden_size,bias=False)
    
    def forward(self,x1,x2):
        '''
        input:
        x1: batch x word_num1 x dim
        '''
        x1=dropout(x1,p=dropout_p,training=self.training)
        x2=dropout(x2,p=dropout_p,training=self.training)

        x1_rep=x1
        x2_rep=x2 

        batch=x1_rep.size(1)
        word_num1=x1_rep.size(1)
        word_num2=x2_rep.size(1)

        dim=x1_rep.size(2)

        if self.correlation_func==2 or self.correlation_func==3:
            x1_rep=self.linear(x1_rep.contiguous.view(-1,dim)).view(batch,word_num1,self.hidden_size)
            x2_rep=self.linear(x2_rep.contiguous.view(-1,dim)).view(batch,word_num2,self.hidden_size)

            if self.correlation_func==3:
                x1_rep=F.relu(x1_rep)            
                x2_rep=F.relu(x2_rep)   

            x1_rep=x1_rep*self.diagonal.expand_as(x1_rep)

        if self.correlation_func==4: 
            x2_rep=self.linear(x2_rep.contiguous())