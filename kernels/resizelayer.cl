const sampler_t s_mySampler = CLK_FILTER_NEAREST  | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE ; 
__kernel void resizelayer(    __read_only image2d_t InBuf,                                                                
                        __write_only image2d_t OutBuf,                                                       
                        __const int imgW,                                                                      
                        __const int imgH,                                                                      
                        __const int OutW,                                                                      
                        __const int OutH)                                                                      
{
    int col = get_global_id(0);                                                                                      
    int row = get_global_id(1);                                                                                      
                                                                                                                     
    float RowPercent = (float)row/(float)OutH ;                                                                         
    float ColPercent = (float)col/(float)OutW ;                                                                         
                                                                                                                     
    int InImgRow = (int) (RowPercent * imgH);                                                                         
    int InImgCol = (int) (ColPercent * imgW);                                                                         
                                                                                                                     
    const int2 pos_in = {InImgCol, InImgRow};                                                                         
    const int2 pos_Out = {col, row};                                                                                 
    int2 pos_Cur;                                                                                                     
    float4 P[6][6];                                                                                                     
    float4 PB[4][4];                                                                                                 
    float4 *Point;                                                                                                     
    float4 pResult[4];                                                                                                 
    float x = 0.5;                                                                                                     
    float4 Out_F;                                                                                                     
    int iRowCount,iColCount;                                                                                         
                                                                                                                     
    P[0][0] = read_imagef(InBuf, s_mySampler, pos_in + (int2)(-2,-2));                                                
    P[0][1] = read_imagef(InBuf, s_mySampler, pos_in + (int2)(-1,-2));                                                
    P[0][2] = read_imagef(InBuf, s_mySampler, pos_in + (int2)( 0,-2));                                                
    P[0][3] = read_imagef(InBuf, s_mySampler, pos_in + (int2)( 1,-2));                                                
    P[0][4] = read_imagef(InBuf, s_mySampler, pos_in + (int2)( 2,-2));                                                
    P[0][5] = read_imagef(InBuf, s_mySampler, pos_in + (int2)( 3,-2));                                                
                                                                                                                    
    P[1][0] = read_imagef(InBuf, s_mySampler, pos_in + (int2)(-2,-1));                                                
    P[1][1] = read_imagef(InBuf, s_mySampler, pos_in + (int2)(-1,-1));                                                
    P[1][2] = read_imagef(InBuf, s_mySampler, pos_in + (int2)( 0,-1));                                                
    P[1][3] = read_imagef(InBuf, pos_in + (int2)( 1,-1));                                                
    P[1][4] = read_imagef(InBuf, pos_in + (int2)( 2,-1));                                                
    P[1][5] = read_imagef(InBuf, pos_in + (int2)( 3,-1));                                                
                                                                                                        
    P[2][0] = read_imagef(InBuf, pos_in + (int2)(-2, 0));                                                
    P[2][1] = read_imagef(InBuf, pos_in + (int2)(-1, 0));                                                
    P[2][2] = read_imagef(InBuf, pos_in + (int2)( 0, 0));                                                
    P[2][3] = read_imagef(InBuf, pos_in + (int2)( 1, 0));                                                
    P[2][4] = read_imagef(InBuf, pos_in + (int2)( 2, 0));                                                
    P[2][5] = read_imagef(InBuf, pos_in + (int2)( 3, 0));                                                
                                                                                                        
    P[3][0] = read_imagef(InBuf, pos_in + (int2)(-2, 1));                                                
    P[3][1] = read_imagef(InBuf, pos_in + (int2)(-1, 1));                                                
    P[3][2] = read_imagef(InBuf, pos_in + (int2)( 0, 1));                                                
    P[3][3] = read_imagef(InBuf, pos_in + (int2)( 1, 1));                                                
    P[3][4] = read_imagef(InBuf, pos_in + (int2)( 2, 1));                                                
    P[3][5] = read_imagef(InBuf, pos_in + (int2)( 3, 1));                                                
                                                                                                        
    P[4][0] = read_imagef(InBuf, pos_in + (int2)(-2, 2));                                                
    P[4][1] = read_imagef(InBuf, pos_in + (int2)(-1, 2));                                                
    P[4][2] = read_imagef(InBuf, pos_in + (int2)( 0, 2));                                                
    P[4][3] = read_imagef(InBuf, pos_in + (int2)( 1, 2));                                                
    P[4][4] = read_imagef(InBuf, pos_in + (int2)( 2, 2));                                                
    P[4][5] = read_imagef(InBuf, pos_in + (int2)( 3, 2));                                                
                                                                                                        
    P[5][0] = read_imagef(InBuf, pos_in + (int2)(-2, 3));                                                
    P[5][1] = read_imagef(InBuf, pos_in + (int2)(-1, 3));                                                
    P[5][2] = read_imagef(InBuf, pos_in + (int2)( 0, 3));                                                
    P[5][3] = read_imagef(InBuf, pos_in + (int2)( 1, 3));                                                
    P[5][4] = read_imagef(InBuf, pos_in + (int2)( 2, 3));                                                
    P[5][5] = read_imagef(InBuf, pos_in + (int2)( 3, 3));                                                
                                                                                                                
   PB[0][0] = (P[0][0] + P[0][1] + P[0][2] + P[1][0] + P[1][1] + P[1][2] + P[2][0] + P[2][1] + P[2][2]) / 9.0f;        
   PB[0][1] = (P[0][1] + P[0][2] + P[0][3] + P[1][1] + P[1][2] + P[1][3] + P[2][1] + P[2][2] + P[2][3]) / 9.0f;        
   PB[0][2] = (P[0][2] + P[0][3] + P[0][4] + P[1][2] + P[1][3] + P[1][4] + P[2][2] + P[2][3] + P[2][4]) / 9.0f;        
   PB[0][3] = (P[0][3] + P[0][4] + P[0][5] + P[1][3] + P[1][4] + P[1][5] + P[2][3] + P[2][4] + P[2][5]) / 9.0f;        
                                                                                                                    
   PB[1][0] = (P[1][0] + P[1][1] + P[1][2] + P[2][0] + P[2][1] + P[2][2] + P[3][0] + P[3][1] + P[3][2]) / 9.0f;        
   PB[1][1] = (P[1][1] + P[1][2] + P[1][3] + P[2][1] + P[2][2] + P[2][3] + P[3][1] + P[3][2] + P[3][3]) / 9.0f;        
   PB[1][2] = (P[1][2] + P[1][3] + P[1][4] + P[2][2] + P[2][3] + P[2][4] + P[3][2] + P[3][3] + P[3][4]) / 9.0f;        
   PB[1][3] = (P[1][3] + P[1][4] + P[1][5] + P[2][3] + P[2][4] + P[2][5] + P[3][3] + P[3][4] + P[3][5]) / 9.0f;        
                                                                                                                    
   PB[2][0] = (P[2][0] + P[2][1] + P[2][2] + P[3][0] + P[3][1] + P[3][2] + P[4][0] + P[4][1] + P[4][2]) / 9.0f;        
   PB[2][1] = (P[2][1] + P[2][2] + P[2][3] + P[3][1] + P[3][2] + P[3][3] + P[4][1] + P[4][2] + P[4][3]) / 9.0f;        
   PB[2][2] = (P[2][2] + P[2][3] + P[2][4] + P[3][2] + P[3][3] + P[3][4] + P[4][2] + P[4][3] + P[4][4]) / 9.0f;        
   PB[2][3] = (P[2][3] + P[2][4] + P[2][5] + P[3][3] + P[3][4] + P[3][5] + P[4][3] + P[4][4] + P[4][5]) / 9.0f;        
                                                                                       
   PB[3][0] = (P[3][0] + P[3][1] + P[3][2] + P[4][0] + P[4][1] + P[4][2] + P[5][0] + P[5][1] + P[5][2]) / 9.0f;      
   PB[3][1] = (P[3][1] + P[3][2] + P[3][3] + P[4][1] + P[4][2] + P[4][3] + P[5][1] + P[5][2] + P[5][3]) / 9.0f;      
   PB[3][2] = (P[3][2] + P[3][3] + P[3][4] + P[4][2] + P[4][3] + P[4][4] + P[5][2] + P[5][3] + P[5][4]) / 9.0f;      
   PB[3][3] = (P[3][3] + P[3][4] + P[3][5] + P[4][3] + P[4][4] + P[4][5] + P[5][3] + P[5][4] + P[5][5]) / 9.0f;      
                                                                                       
   Point = PB[0];                                                                           
   pResult[0] = Point[1] + 0.5f * x*(Point[2] - Point[0] + x*(2.0f*Point[0] - 5.0f*Point[1] + 4.0f*Point[2] - Point[3] + x*(3.0f*(Point[1] - Point[2]) + Point[3] - Point[0])));   
   Point = PB[1];                                                                                                                        
   pResult[1] = Point[1] + 0.5f * x*(Point[2] - Point[0] + x*(2.0f*Point[0] - 5.0f*Point[1] + 4.0f*Point[2] - Point[3] + x*(3.0f*(Point[1] - Point[2]) + Point[3] - Point[0])));   
   Point = PB[2];                                                                                                                        
   pResult[2] = Point[1] + 0.5f * x*(Point[2] - Point[0] + x*(2.0f*Point[0] - 5.0f*Point[1] + 4.0f*Point[2] - Point[3] + x*(3.0f*(Point[1] - Point[2]) + Point[3] - Point[0])));   
   Point = PB[3];                                                                                                                        
   pResult[3] = Point[1] + 0.5f * x*(Point[2] - Point[0] + x*(2.0f*Point[0] - 5.0f*Point[1] + 4.0f*Point[2] - Point[3] + x*(3.0f*(Point[1] - Point[2]) + Point[3] - Point[0])));   
   Point = pResult;                                                                                                                     
   Out_F = Point[1] + 0.5f * x*(Point[2] - Point[0] + x*(2.0f*Point[0] - 5.0f*Point[1] + 4.0f*Point[2] - Point[3] + x*(3.0f*(Point[1] - Point[2]) + Point[3] - Point[0])));          
   write_imagef(OutBuf, pos_Out,Out_F);                                                                                      
}
