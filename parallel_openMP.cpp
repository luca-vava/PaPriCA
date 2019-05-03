#include <omp.h>          
#include <stdio.h>           
#include <stdlib.h>        
#include <math.h>         
#include <ctime>

void scan(int size, int threads, int out[]);
void out_print(int dim, double elapsed_secs);

int main()
{	
 	using namespace std;

	for(int k = 10; k<= 29; k++) 				//iterazione per test delarray sulle k diverse potenze di 2
	{ 
 		int threads=14; 				//# di threads sui quali far girare il metodo di scan
 		int size = pow(2, k); 
		double elapsed_secs = 0 ;


		for (int i = 0; i<30; i++) 			//iterazione per calcolare un tempo medio su 30 cicli della procedura di scan
		{

			int *out = new int[size];		//allocazione array nell'heap
			srand (time(NULL)); 			//randomizzazione del seed

			for (int j = 0; j < size; j++)
			{
				out[j] = rand()%100; 		// generazione numeri pseudocasuali 
			}
						 
			double start_time = omp_get_wtime(); 	//calcolo e somma tempi esecuzione scan 	
			scan(size, threads, out);
			elapsed_secs += omp_get_wtime() - start_time;

			start_time = 0;                       
			delete [] out;

		}

		printf("%d, %f\n", k, elapsed_secs);
		out_print(k, elapsed_secs/30);
	}

	return 0;
}

void scan(int size, int threads, int out[])
{   
	int offset = 0;
	int tot = 0;
	int op_step = size/2;

//////////////// downsweep phase

    //printf("-----------------DOWN----------------\n");
	
	for (int k = 1; op_step >= 1; k++) 
	{
		offset = pow(2, k)-1;   
		tot = pow(2, k-1);

	    #pragma omp parallel for num_threads(threads) schedule(static) 
	    for (int i = offset; i <= size; i += offset+1)
	    {
	    	out[i] += out[i - tot];        
	    }
			
		op_step /= 2;
		#pragma omp barrier
	}

	out[size-1] = 0;
    //printf("-----------------END - DOWN----------------\n");

///////////////// upsweep phase
    //printf("-----------------UP----------------\n");
	for (int k=(size/2); k>0; k /=2)
	{
		#pragma omp parallel for num_threads(threads) schedule(static) 
		for (int i = 0; i<size; i+=k*2)
		{
			int tmp=out[i+k-1];
			out[i+k-1] = out[i+k*2 -1];
			out[i+k*2 -1] = tmp + out[i+k*2 -1];
		}
		#pragma omp barrier
	}

	printf("-----------------END - UP----------------\n");
}

void out_print(int dim, double elapsed_secs)
{
	FILE *outfile; 
	outfile = fopen("out.txt", "a");
	fprintf(outfile, "dim: %d, sec: %f\n", dim, elapsed_secs);
	fclose(outfile);
}

