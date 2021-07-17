#include <stdio.h>
#include <stdlib.h>
#include <omp.h> 
#include "DS_definitions.h"
#include "DS_timer.h"

void merge(int* a, int* b, int left, int mid, int right);
void merge_sort(int* a, int* b, int left, int right);

int main(int argc, char *argv[]) {
	DS_timer timer(2);

	timer.setTimerName(0, (char*)"Merge Sort(Serial)");
	timer.setTimerName(1, (char*)"Merge Sort(Parallel)");
	
	int n = atoi(argv[1]); // 배열의 크기
	int p = omp_get_num_threads(); // 쓰레드의 개수
	int m = n / p;

	const int SIZE = n * sizeof(int); 

	int* a = (int*)malloc(SIZE);
	int* b = (int*)malloc(SIZE);

	int* serial_result = (int*)malloc(SIZE);
	int* parallel_result = (int*)malloc(SIZE);

	printf("Size of Array : %d", n);
	printf("\n");
	printf("\n");

	int i;
	
	// 배열에 1~500 랜덤값 넣기
	for (i = 0; i < n; i++) {
		a[i] = rand() % 500 + 1;
	}
	
	// 배열 프린트
	printf("Print Array\n");

	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
	}
	printf("\n");
	printf("\n");

	// Serial Merge Sort
	timer.onTimer(0);

	merge_sort(a, b, 0, n-1);

	printf("Print Serial Merge Sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		serial_result[i] = a[i];
	}
	printf("\n");
	printf("\n");

	timer.offTimer(0);

	// Parallel Merge Sort
	timer.onTimer(1);

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int start = m * tid;
		int end = m * (tid + 1);

		if (tid < p - 1) // 마지막 쓰레드 빼고 다
			merge_sort(a, b, start, end);
		else // 마지막 쓰레드
			merge_sort(a, b, start, n - 1);
	}

	for (i = 1; i < p - 1; i++) { // 마지막 쓰레드 빼고 다
		int middle = i * m;
		merge(a, b, 0, middle, middle + m - 1);
	}

	merge(a, b, 0, (p - 1) * m, n - 1); // 마지막 쓰레드

	printf("Print Parallel Merge Sort\n");
	for (i = 0; i < n; i++) {
		printf("%d ", a[i]);
		parallel_result[i] = a[i];
	}
	printf("\n");
	printf("\n");

	timer.offTimer(1);

	// Serial결과와 Parallel결과 비교
	bool isCorrect = true;

	for (int i = 0; i < n; i++) {
		if (serial_result[i] != parallel_result[i]) {
			isCorrect = false;
			break;
		}
	}

	if (isCorrect)
		printf("Results are matched! :)\n");
	else
		printf("Results are not matched :(\n");

	timer.printTimer();
	EXIT_WIHT_KEYPRESS;
}

// 합병 정렬
void merge_sort(int* a, int* b, int left, int right) {
	int mid;
	if (right > left) {
		mid = (right + left) / 2; 
		merge_sort(a, b, left, mid); // 앞부분 리스트 정렬
		merge_sort(a, b, mid + 1, right); // 뒷부분 리스트 정렬
		merge(a, b, left, mid + 1, right); // 정렬된 2개의 배열 합병
	}
}

// 2개의 정렬된 배열 합병
void merge(int* a, int* b, int left, int mid, int right) {
	int i, left_end, count;
	int j;
	left_end = mid - 1;
	j = left;
	count = right - left + 1;

	// 분할 정렬된 배열 합병
	while ((left <= left_end) && (mid <= right)) {
		if (a[left] <= a[mid]) {
			b[j] = a[left];
			j++;
			left++;
		}
		else {
			b[j] = a[mid];
			j++;
			mid++;
		}
	}

	// 왼쪽에 남아 있는 값 복사
	while (left <= left_end) {
		b[j] = a[left];
		left++;
		j++;
	}

	// 오른쪽에 남아 있는 값 복사
	while (mid <= right) {
		b[j] = a[mid];
		mid++;
		j++;
	}

	// 원래 배열로 재복사
	for (i = 0; i < count; i++) {
		// for (i = 0; i <= count; i++) {
		a[right] = b[right];
		right--;
	}
}
