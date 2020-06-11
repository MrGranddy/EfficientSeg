#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct instruction_s{
    int step, index;
} instruction_s;


char* get_word(char* s, int n){
    int start_idx = -1;
    int end_idx = -1;
    int space_cnt = 0;
    int i;
    for(i = 0; i < strlen(s); i++){
        if( s[i] == ' ' ){
            space_cnt += 1;
        }
        if( (i == 0 || s[i] == ' ') && space_cnt == n && start_idx == -1 ){
            start_idx = i+1;
            if(i == 0) start_idx--;
        }
        if( s[i] == ' '  && space_cnt == n+1 && end_idx == -1 ){
            end_idx = i-1;
        }
    }
    if(end_idx == -1) end_idx = strlen(s) - 1;
    char *word = (char *)malloc(sizeof(char) * (end_idx - start_idx + 2));
    for(i = start_idx; i <= end_idx; i++){
        word[i - start_idx] = s[i];
    }
    word[end_idx - start_idx + 1] = 0;
    return word;
}

void get_sentence(char** lines, struct instruction_s* instructions, int n_instructions, char* sentence){
    int idx, step;

    int offset = 0;
    int s_ptr = 0;

    int i, j;

    char *word;
    char *line;

    for( i = 0; i < n_instructions; i++ ){
        idx = instructions[i].index;
        step = instructions[i].step;

        offset += step;
        line = lines[offset-1];
        word = get_word(line, idx-1);

        for(j = 0; j < strlen(word); j++){
            sentence[s_ptr++] = word[j];
        }
        if( i != n_instructions -1 ) sentence[s_ptr++] = ' ';

        free(word);
    }
}


int main( int argc, char **argv ){

    const char *book_path = argv[1];
    const char *ins_path = argv[2];
    
    size_t MAX_BOOK_LINES = 10000;
    size_t MAX_LINE_SIZE = 200;
    size_t MAX_INS_SIZE = 1000;

    size_t real_num_book_lines;
    size_t real_line_size;
    size_t real_ins_size;

    char **tmp_book;
    char *tmp_line;
    instruction_s *tmp_ins;

    char **book;
    instruction_s *instructions;

    FILE *fid_book;
    FILE *fid_ins;

    int i, j;

    /* Book reading phase start */
    book = (char **)malloc(sizeof(char *) * MAX_BOOK_LINES);
    for(i = 0; i < MAX_BOOK_LINES; i++){
        book[i] = (char *)malloc(sizeof(char) * MAX_LINE_SIZE);
    } // allocate memory for the book

    fid_book = fopen(book_path, "r"); // open book fid
    for(i = 0; !feof(fid_book); i++){
        //getline(book + i, &MAX_LINE_SIZE, fid_book);
        fgets(book[i], MAX_LINE_SIZE, fid_book);
    }
    fclose(fid_book);
    real_num_book_lines = i - 1;

    tmp_book = (char **)malloc(sizeof(char *) * real_num_book_lines); //reallocate for better fit to memory
    for(i = 0; i < real_num_book_lines; i++){
        real_line_size = strlen(book[i]);
        tmp_line = (char *)malloc(sizeof(char) * (real_line_size + 1));
        strcpy(tmp_line, book[i]);
        tmp_line[real_line_size - 2] = 0; // last char
        tmp_line[real_line_size - 1] = 0; // newline make it 0
        tmp_book[i] = tmp_line; // add to array
    }
    for(i = 0; i < MAX_BOOK_LINES; i++) free(book[i]);
    free(book); // delete old
    book = tmp_book;
    // all above are copying to a better fit array
    /* Book reading phase over */



    /* Instruction reading phase start */
    tmp_ins = (instruction_s *)malloc(sizeof(instruction_s) * MAX_INS_SIZE);
    fid_ins = fopen(ins_path, "r");

    for( i = 0; !feof(fid_ins); i++){
        fscanf(fid_ins, "%d %d", &tmp_ins[i].step, &tmp_ins[i].index);
    }
    fclose(fid_ins);
    real_ins_size = i - 1;
    instructions = (instruction_s *)malloc(sizeof(instruction_s) * real_ins_size);
    for( i = 0; i < real_ins_size; i++ ){
        instructions[i].step = tmp_ins[i].step;
        instructions[i].index = tmp_ins[i].index;
    }
    free(tmp_ins);
    /* Instruction reading phase over */

    char *sentence = (char *)malloc(sizeof(char) * MAX_LINE_SIZE);

    get_sentence(book, instructions, real_ins_size, sentence);
    printf("%s\n", sentence);

    free(instructions);
    for(i = 0; i < real_num_book_lines; i++) free(book[i]);
    free(book); // free allocated memory

    return 0;

}