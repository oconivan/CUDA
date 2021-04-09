//
// Lab 2
// Author: Ivan Ocon
// Date: 09/15/2019
//

#include <stdio.h>			// Standard I/O library
#include <unistd.h>			// UNIX system call library
#include <stdlib.h>			// Standard library: atol()
#include <signal.h>         // Signal handler library
#include <pthread.h>		// pthreads library
#include <string.h>
#include <time.h>

pthread_t tid1, tid2, tid3, tid4;
pthread_attr_t attr;
long double c=1;
long int count = 1;
long int previous;
long int catalan_numbers;
long int thread_numbers;

int p;

void *catalan(void *n)
{
	     FILE *catalanFile;
		 //long double largest; //to get the largest possible range possible
		 
		 char str[50];
		sprintf(str, "Catalan_%d.dat", p);
		//sprintf(str, "%c", ".dat");
		//str2 = strcat(str2, ".dat");
		
		 catalanFile = fopen(str,"wb");

		 int i;
		
		printf("This is thread: %d\n", p);
		 
         for(i=previous;i< (long int)n;i++)
         {
                 c=(2*(2*i-1)*c)/(i+1);
				 fprintf(catalanFile, "%lu = %Lf\n", i, c);
				/* Call the catalan numbers to print out in the screen*/				 
				printf("%d = %Lf\n",i, c);
         }
		 fclose(catalanFile);
		 pthread_exit(0);
}





int main(int argc, char **argv)
{
	pid_t firstPid;
	int how_many_numbers_per_thread;
	int how_many_numbers_per_remainder;
	catalan_numbers = atol(argv[1]);
	thread_numbers = atol(argv[2]);
	double test;

        time_t start, end;
        double cpu_time_used;

        start = time(NULL);
	
    if(argc < 1)
    {
        printf("Usage: pthreadExample num\n");
        return -1;
    }

    /* Set pthread attributes to defaults */
        pthread_attr_init(&attr);
	how_many_numbers_per_thread = catalan_numbers / thread_numbers;
	how_many_numbers_per_remainder = catalan_numbers % thread_numbers;
	
    /* Setup a thread to compute the Catalan numbers */
	for (p=1; p <= thread_numbers; p++)
	{
		previous = count;
		count = count + (long int) how_many_numbers_per_thread;
		//printf("Numbers per thread: %f\n", how_many_numbers_per_thread);
		
		switch (p)
		{
			case 1:
				pthread_create(&tid1,&attr,catalan,(void *)count);
				pthread_join(tid1,NULL);
				break;
			case 2:
			    
				pthread_create(&tid2,&attr,catalan,(void *)count);
				pthread_join(tid2,NULL);
				break;
			case 3:
				pthread_create(&tid3,&attr,catalan,(void *)count);
				pthread_join(tid3,NULL);
				break;
			case 4:
			    count = count + how_many_numbers_per_remainder;
				pthread_create(&tid4,&attr,catalan,(void *)count);
				pthread_join(tid4,NULL);
				break;
			default:
				printf("error");
		//pthread_create(&tid1,&attr,catalan,(void *)atol(argv[1]));
		}
	}
	
    /* Wait for the fibonacci thread to complete */
    //pthread_join(tid1,NULL);
	
	/* fork a child process */
  firstPid = fork();

     if(firstPid < 0)
    {
		/* Negative process id means there was an error */
		fprintf(stderr,"Error forking a process\n");
		return -1;
	}
	else if(firstPid == 0)
	{
		execlp("echo", "echo", "concacatenation from Catalan files will be done in Concatenation.dat file:", NULL);
		//execlp("cat","cat","Catalan_1.dat > catalan.dat", NULL);
		//execlp("cat","cat","Catalan_2.dat", NULL);
		//execlp("cat","cat","Catalan_3.dat", NULL);
		//execlp("cat","cat","Catalan_4.dat", NULL);
		exit(0);
	}
	else
	{
		wait(NULL);
		/* process id is non-zero (i.e., the child id), this is the parent process */
		printf("I am the parent1(pid=%d)\n", getpid());
	}
	

   end = time(NULL);
   printf("%lf seconds \n", end-start);



	// Open the files to be merged 
   FILE *fp1 = fopen("Catalan_1.dat", "rb"); 
   FILE *fp2 = fopen("Catalan_2.dat", "rb"); 
   FILE *fp3 = fopen("Catalan_3.dat", "rb"); 
   FILE *fp4 = fopen("Catalan_4.dat", "rb"); 
  
  
   // Open file to store the result 
   FILE *fp5 = fopen("Concatenation.dat", "wb"); 
   char ch; 
   //char *ch[20]; 
  
   if (fp1 == NULL || fp2 == NULL || fp3 == NULL || fp4 == NULL) 
   { 
         puts("Could not open files"); 
         exit(0); 
   } 
  
   // Copy contents of first file to file5.txt 
   while ((ch = fgetc(fp1)) != EOF) 
      fputc(ch, fp5);  
      ch = fgetc(fp1);
      //printf("%c",ch);	  
      //fprintf(fp5,"%Lf",&fp1);
  
   // Copy contents of second file to file5.txt 
   while ((ch = fgetc(fp2)) != EOF) 
      fputc(ch, fp5); 
      ch = fgetc(fp2);
      //fprintf(fp5,"%Lf",&fp2);
    
     // Copy contents of third file to file5.txt 
   while ((ch = fgetc(fp3)) != EOF) 
      fputc(ch, fp5);
      ch = fgetc(fp3);  
      //fprintf(fp5,"%Lf",&fp3);
    
     // Copy contents of fourth file to file5.txt 
   while ((ch = fgetc(fp4)) != EOF) 
      fputc(ch, fp5); 
      ch = fgetc(fp4);
      //fprintf(fp5,"%Lf",&fp4);
   
  
   printf("Concatenatenated files done\n"); 
  
   fclose(fp1); 
   fclose(fp2); 
   fclose(fp3);
   fclose(fp4);
   fclose(fp5);




	
    /* exit program */
    return 0;

}
