//状態はその行動に合わせられると定義(行動0→状態0，行動1→状態1へ遷移)

//乱数マクロ
#define rand_int() (genrand_int32())
#define rand_double() (genrand_real3())

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mt19937ar.h"

//Q関連
#define NumofEpoch 3000
#define LearningRateOfQ 0.1
#define gamma 0.9
//epsilon-greedy行動選択
#define epsilon 0.5
//NN周り
#define InputNum 2			//入力層サイズ
#define HiddenNum 2			//中間層サイズ
#define OutputNum 2			//出力層サイズ
#define LearningRateOfNN 3	//NNの学習率

double NNWeightOfInputHidden[InputNum+1][HiddenNum];		//そのUEに対しての入力-隠れ層の重み
double NNWeightOfHiddenOutput[HiddenNum+1][OutputNum];		//そのUEに対しての隠れ-出力層の重み
double outputOfHidden[HiddenNum];							//そのUEが持っている隠れ層の出力
double outputOfOutput[OutputNum];							//そのUEが持っている出力層の出力

const double R[2][2] = {{0.5,0.2},
					    {0.2,0.5}};

//Q
int take_action(int state);		 										//行動決定
double get_reward(int state,int action){return R[state][action];};		//報酬獲得
void update_Q(double Q[OutputNum],int action,double reward);			//Q値更新
//NN
void make_feature(int state,int out[InputNum]);						//特徴量ベクトル生成
void foward_calc(int in[InputNum],double out[OutputNum]);			//順方向Q計算
void backward_learn(double Q[OutputNum], int state, int action);	//逆方向学習
double sigmoid(double u){return 1.0/(1.0 + exp(-u));}				//シグモイド関数
void NN_init();														//NN初期化

int main() {
	int action,state;//行動と状態
	double reward;
	int t;
	double Q[OutputNum];
	int tempIn[InputNum];
	//データ取り用
	double tempOut[OutputNum];
	int i,j;

	init_genrand((unsigned)time(NULL));
	NN_init();
	state = rand_int() % InputNum;

	for(t = 0; t < NumofEpoch; t++) {
		action = take_action(state);//状態から行動を決定
		reward = get_reward(state,action);//状態と行動から報酬を獲得

		//Q値をもう一回求めて報酬を更新
		make_feature(state,tempIn);
		foward_calc(tempIn,Q);
		update_Q(Q,action,reward);

		backward_learn(Q,state, action);//BP

		printf("%d\t",t);
		for(i=0;i<InputNum;i++){
			make_feature(i,tempIn);
			foward_calc(tempIn,tempOut);
			printf("%f\t%f\t",tempOut[0],tempOut[1]);
		}
		printf("\n");


		state = action;
	}
	return 0;
}

int take_action(int state) {
	int out;
	int i;
	double Q[OutputNum];
	int in[InputNum];
	double temp=-INFINITY;

	if(genrand_real3() < epsilon){
		out = genrand_int32() % 2;
	}else{
		make_feature(state,in);
		foward_calc(in,Q);//NNでQ値もらう
		out = 0;
		for(i=0;i<OutputNum;i++){
			if(temp < Q[i]){//Qが今見てるものよりデカイ
				temp = Q[i];
				out = i;
			}
		}
	}
	return out;
}

void update_Q(double Q[OutputNum],int action,double reward){
	Q[action] = ((1.0-LearningRateOfQ) * Q[action]) + (LearningRateOfQ * reward);
	return;
}

void make_feature(int state,int out[InputNum]){
	int i;

	for(i=0;i<InputNum;i++){
		out[i] = 0;
	}
	out[state] = 1;
	return;
}

void foward_calc(int in[InputNum],double out[OutputNum]){
	int i,j;

	//入力-隠れそう計算
	for(i=0;i<HiddenNum;i++){
		outputOfHidden[i] = 0;
		for(j=0;j<InputNum;j++){
			outputOfHidden[i] += (NNWeightOfInputHidden[j][i] * ((double)in[j]));
		}
		outputOfHidden[i] -= NNWeightOfInputHidden[InputNum][i];//しきい値
		outputOfHidden[i] = sigmoid(outputOfHidden[i]);
	}
	//隠れ-出力層計算
	for(i=0;i<OutputNum;i++){
		outputOfOutput[i] = 0;
		for(j=0;j<HiddenNum;j++){
			outputOfOutput[i] += (NNWeightOfHiddenOutput[j][i] * outputOfHidden[j]);
		}
		outputOfOutput[i] -= NNWeightOfHiddenOutput[HiddenNum][i];
		outputOfOutput[i] = sigmoid(outputOfOutput[i]);
		out[i] = outputOfOutput[i];
	}

	return;
}

void backward_learn(double Q[OutputNum],int state,int action){
	int i,j;
	double temp,error;
	int in[InputNum];

	//入力情報作成
	make_feature(state,in);
	
	//隠れ-出力層の重み学習
	error = Q[action] - outputOfOutput[action];
	for(j=0;j<HiddenNum;j++){
		NNWeightOfHiddenOutput[j][action] += LearningRateOfNN * error * outputOfOutput[action] * (1.0 - outputOfOutput[action]) * outputOfHidden[j];
	}
	//しきい値分
	NNWeightOfHiddenOutput[j][action] += LearningRateOfNN * error * outputOfOutput[action] * (1.0 - outputOfOutput[action]) * (-1.0);
	//入力-隠れの重み学習
	for(i=0;i<HiddenNum;i++){
		temp = outputOfHidden[i] * (1.0 - outputOfHidden[i]) * NNWeightOfHiddenOutput[i][action] * error * outputOfOutput[action] * (1.0 - outputOfOutput[action]);
		for(j=0;j<InputNum;j++){
			NNWeightOfInputHidden[j][i] += LearningRateOfNN * (double)in[j] * temp;
		}
		NNWeightOfInputHidden[j][i] += LearningRateOfNN * -1.0 * temp;
	}
	return;
}

void NN_init(){
	int i,j;
	for(i=0;i<=InputNum;i++){
		for(j=0;j<HiddenNum;j++){
			NNWeightOfInputHidden[i][j] = rand_double();	//入力-隠れの初期化
		}
	}
	for(i=0;i<=HiddenNum;i++){
		for(j=0;j<OutputNum;j++){
			NNWeightOfHiddenOutput[i][j] = rand_double();	//隠れ-出力の初期化
		}
	}

	return;
}
