#include <omp.h>          
#include <stdio.h>           
#include <stdlib.h>        
#include <math.h>         
#include <ctime>


//void scan(int size, int threads, int out[]);
void out_print(int dim, double elapsed_secs, char *type);

template<typename T> 
class scan{

	int offset;
	int tot;

//////////////// downsweep phase

    //printf("-----------------DOWN----------------\n");
public: 
	scan(int size, int threads, T out[])
	{
		int op_step = size/2;

		for (int k = 1; op_step >= 1; k++) 
		{
			offset = pow(2, k)-1;   
			tot = pow(2, k-1);

			#pragma omp parallel for num_threads(threads) ordered schedule(static) 
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
			#pragma omp parallel for schedule(static) num_threads(threads)
			for (int i = 0; i<size; i+=k*2)
			{
				int tmp=out[i+k-1];
				out[i+k-1] = out[i+k*2 -1];
				out[i+k*2 -1] = tmp + out[i+k*2 -1];
			}
			#pragma omp barrier
		}
		
	}
};

int main()
{	
 	using namespace std;

//////// TEST DIMENSIONE ///////////
	for(int k = 10; k<= 29; k++)
	{ 
 		//int threads = 14; // inserire procedura automatica che
 						// riconosce quanti threads possono essere creati
		int threads = omp_get_num_procs();
 		int size = pow(2, k); 
		double elapsed_secs = 0 ;
		double elapsed_secs_short = 0;
		double elapsed_secs_Ushort = 0;
		double elapsed_secs_int = 0;
		double elapsed_secs_Uint = 0;
		double elapsed_secs_long = 0;
		double elapsed_secs_Ulong = 0;
		double elapsed_secs_longlong = 0;
		double elapsed_secs_Ulonglong = 0;
		double elapsed_secs_float = 0;
		double elapsed_secs_double = 0;
		double elapsed_secs_long_double = 0;


		for (int i = 0; i<30; i++)
		{
			//allocazione array nell'heap

			short *out_short = new short[size];
			unsigned short *out_Ushort = new unsigned short[size];

			int *out_int = new int[size]; 
			unsigned int *out_Uint = new unsigned int[size];
			
			long *out_long = new long[size];
			unsigned long *out_Ulong = new unsigned long[size];

			long long *out_longlong = new long long[size];
			unsigned long long *out_Ulonglong = new unsigned long long[size];

			float *out_float = new float[size];
			double *out_double = new double[size];
			long double *out_long_double = new long double[size];

			//srand (time(NULL));

			for (int j = 0; j < size; j++)
			{
				out_short[j] = (short)rand();
				out_Ushort[j] = (unsigned short)rand();

				out_int[j] = (int)rand();
				out_Uint[j] = (unsigned int)rand();

				out_long[j] = (long)rand();
				out_Ulong[j] = (unsigned long)rand();

				out_longlong[j] = (long long)rand();
				out_Ulonglong[j] = (unsigned long long)rand();

				out_float[j] = (float)rand();
				out_double[j] = (double)rand();
				out_long_double[j] = (long double)rand();
			}


////////////TEST TIPO DI DATO IN SINGOLA E DOPPIA PRECISIONE ////////////////

			double start_time_short = omp_get_wtime();
			scan<short>(size, threads, out_short);
			elapsed_secs_short += omp_get_wtime() - start_time_short;
			delete [] out_short;

			double start_time_Ushort = omp_get_wtime();
			scan<unsigned short>(size, threads, out_Ushort);
			elapsed_secs_Ushort += omp_get_wtime() - start_time_Ushort;
			delete [] out_Ushort;


			
			double start_time_int = omp_get_wtime();
			scan<int>(size, threads, out_int);
			elapsed_secs_int += omp_get_wtime() - start_time_int;
			delete [] out_int;

			double start_time_Uint = omp_get_wtime();
			scan<unsigned int>(size, threads, out_Uint);
			elapsed_secs_Uint += omp_get_wtime() - start_time_Uint;
			delete [] out_Uint;



			double start_time_long = omp_get_wtime();	
			scan<long>(size, threads, out_long);
			elapsed_secs_long += omp_get_wtime() - start_time_long;
			delete [] out_long;

			double start_time_Ulong = omp_get_wtime();
			scan<unsigned long>(size, threads, out_Ulong);
			elapsed_secs_Ulong += omp_get_wtime() - start_time_Ulong;
			delete [] out_Ulong;



			double start_time_longlong = omp_get_wtime();
			scan<long long>(size, threads, out_longlong);
			elapsed_secs_longlong += omp_get_wtime() - start_time_longlong;
			delete [] out_longlong;

			double start_time_Ulonglong = omp_get_wtime();
			scan<unsigned long long>(size, threads, out_Ulonglong);
			elapsed_secs_Ulonglong += omp_get_wtime() - start_time_Ulonglong;
			delete [] out_Ulonglong;



			double start_time_float = omp_get_wtime();
			scan<float>(size, threads, out_float);
			elapsed_secs_float += omp_get_wtime() - start_time_float;
			delete[] out_float;
			
			double start_time_double = omp_get_wtime();
			scan<double>(size, threads, out_double);
			elapsed_secs_double += omp_get_wtime() - start_time_double;
			delete[] out_double;			

			double start_time_long_double = omp_get_wtime();
			scan<long double>(size, threads, out_long_double);
			elapsed_secs_long_double += omp_get_wtime() - start_time_long_double;
			delete[] out_long_double;

			printf("%d, %d\n", k, i);



			//elapsed_secs += omp_get_wtime() - start_time;

			//start_time = 0;                       
		}

		//printf("%d, %f\n", k, elapsed_secs/30);
		elapsed_secs = elapsed_secs_short;
		out_print(k, elapsed_secs/30, "short");
		elapsed_secs = elapsed_secs_Ushort;
		out_print(k, elapsed_secs/30, "Ushort");
		elapsed_secs = elapsed_secs_int;
		out_print(k, elapsed_secs/30, "int");
		elapsed_secs = elapsed_secs_Uint;
		out_print(k, elapsed_secs/30, "Uint");
		elapsed_secs = elapsed_secs_long;
		out_print(k, elapsed_secs/30, "long");
		elapsed_secs = elapsed_secs_Ulong;
		out_print(k, elapsed_secs/30, "Ulong");
		elapsed_secs = elapsed_secs_longlong;
		out_print(k, elapsed_secs/30, "long_long");
		elapsed_secs = elapsed_secs_Ulonglong;
		out_print(k, elapsed_secs/30, "Ulong_long");
		elapsed_secs = elapsed_secs_float;
		out_print(k, elapsed_secs/30, "float");
		elapsed_secs = elapsed_secs_double;
		out_print(k, elapsed_secs/30, "double");
		elapsed_secs = elapsed_secs_long_double;
		out_print(k, elapsed_secs/30, "long_double");
	}

	return 0;
}

void out_print(int dim, double elapsed_secs, char *type)
{
	FILE *outfile; 
	outfile = fopen("out.txt", "a");
	fprintf(outfile, "dim: %d, type: %s, sec: %f\n", dim, type, elapsed_secs);
	fclose(outfile);
}

