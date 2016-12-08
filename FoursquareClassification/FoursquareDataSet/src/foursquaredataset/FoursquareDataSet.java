
package foursquaredataset;


import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;

import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;


public class FoursquareDataSet {

	public static String[][] readXLSFile() throws IOException
	{
            
                String[][] foursquareDataEntry = new String[3379][7];
                int i=0; 
                int j=0;
                    
                InputStream ExcelFileToRead = new FileInputStream("E:/Foursquare.xls");
		HSSFWorkbook wb = new HSSFWorkbook(ExcelFileToRead);

		HSSFSheet sheet=wb.getSheetAt(0);
		HSSFRow row; 
		HSSFCell cell;

		Iterator rows = sheet.rowIterator();
                
              
                while(rows.hasNext()){

                    row=(HSSFRow) rows.next();
                    
                    for(j =0; j<7; j++){
                                cell = row.getCell(j);


                                if (cell.getCellType() == HSSFCell.CELL_TYPE_STRING)
                                {
                                    foursquareDataEntry[i][j]= cell.getStringCellValue();
                                    System.out.print(cell.getStringCellValue()+"   ");
                                   
                                }
                                else if(cell.getCellType() == HSSFCell.CELL_TYPE_NUMERIC)
                                {
                                    foursquareDataEntry[i][j]=Double.toString(cell.getNumericCellValue());
                                    System.out.print(cell.getNumericCellValue()+"   ");
                                    
                                }
                                else
                                {
                                    //U Can Handel Boolean, Formula, Errors
                                    System.out.println("FFUUCCKK   ");
                                   
                                }
                    }
                  
                    i++;
                    System.out.println(" ");
                }
                return foursquareDataEntry;
	}
	
	public static void writeXLSFile(String[][] foursquareDataEntry) throws IOException {
		
		String excelFileName = "E:/Categories.xls";//name of excel file

		String sheetName = "Sheet1";//name of sheet

		HSSFWorkbook wb = new HSSFWorkbook();
		HSSFSheet sheet = wb.createSheet(sheetName) ;

		//iterating r number of rows
		for (int r=0;r < 3379; r++ )
		{
			HSSFRow row = sheet.createRow(r);
	
			//iterating c number of columns
			for (int c=0;c < 7; c++ )
			{
				HSSFCell cell = row.createCell(c);
				
				cell.setCellValue(foursquareDataEntry[r][c]);
			}
                        
                        String fourSquareType = foursquareDataEntry[r][3];
                        
                        if(r<3861)
                        if(fourSquareType.contains("Ministr")){
                                System.out.println("Administration");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Administration");
                        }
                        
                        else if(fourSquareType.contains("Department")){
                                System.out.println("Administration");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Administration");
                        }
                        
                        else if(fourSquareType.contains("Police Station")){
                                System.out.println("Administration");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Administration");
                        }
                        
                        else if(fourSquareType.contains("Authorit")){
                                System.out.println("Administration");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Administration");
                        }
                        
                        else if(fourSquareType.contains("Defense")){
                                System.out.println("Administration");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Administration");
                        }
                        
                        else if(fourSquareType.contains("Court")){
                                System.out.println("Administration");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Administration");
                        }
                        
                        else if(fourSquareType.contains("School")){
                                System.out.println("Education");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Education");
                        }
                        
                        else if(fourSquareType.contains("College")){
                                System.out.println("Education");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Education");
                        }
                        
                        else if(fourSquareType.contains("University")){
                                System.out.println("Education");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Education");
                        }
                        
                        else if(fourSquareType.contains("Cinema")){
                                System.out.println("Recreation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Recreation");
                        }
                        
                        else if(fourSquareType.contains("Recreation")){
                                System.out.println("Recreation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Recreation");
                        }
                        
                        else if(fourSquareType.contains("Sport")){
                                System.out.println("Recreation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Recreation");
                        }
                        
                        else if(fourSquareType.contains("Bridge")){
                                System.out.println("Recreation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Recreation");
                        }
                        
                        else if(fourSquareType.contains("Playground")){
                                System.out.println("Recreation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Recreation");
                        }
                        
                        else if(fourSquareType.contains("Dive Spot")){
                                System.out.println("Recreation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Recreation");
                        }
                        
                        else if(fourSquareType.contains("Gym")){
                                System.out.println("Recreation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Recreation");
                        }
                        
                        else if(fourSquareType.contains("Pool")){
                                System.out.println("Recreation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Recreation");
                        }
                        
                        else if(fourSquareType.contains("Park")){
                                System.out.println("Recreation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Recreation");
                        }
                        
                        else if(fourSquareType.contains("Bus Line")){
                                System.out.println("Transportation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Transportation");
                        }
                        
                        else if(fourSquareType.contains("Bus Station")){
                                System.out.println("Transportation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Transportation");
                        }
                        
                        else if(fourSquareType.contains("Road")){
                                System.out.println("Transportation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Transportation");
                        }
                        
                        else if(fourSquareType.contains("Train Station")){
                                System.out.println("Transportation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Transportation");
                        }
                        
                        else if(fourSquareType.contains("Travel")){
                                System.out.println("Transportation");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Transportation");
                        }
                        
                        
                        else if(fourSquareType.contains("Hospital")){
                                System.out.println("Hospitals");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hospitals");
                        }
                        
                                 
                        else if(fourSquareType.contains("Bank")){
                                System.out.println("Commercial & Mercantile");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Commercial & Mercantile");
                        }
                        
                        else if(fourSquareType.contains("Fishing Store")){
                                System.out.println("Commercial & Mercantile");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Commercial & Mercantile");
                        }
                        
                        else if(fourSquareType.contains("Shopping Mall")){
                                System.out.println("Commercial & Mercantile");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Commercial & Mercantile");
                        }
                        
                        else if(fourSquareType.contains("Store")){
                                System.out.println("Commercial & Mercantile");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Commercial & Mercantile");
                        }
                        
                        else if(fourSquareType.contains("market")){
                                System.out.println("Commercial & Mercantile");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Commercial & Mercantile");
                        }
                        
                        else if(fourSquareType.contains("Wholesale")){
                                System.out.println("Commercial & Mercantile");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Commercial & Mercantile");
                        }
                        
                        else if(fourSquareType.contains("Retail")){
                                System.out.println("Commercial & Mercantile");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Commercial & Mercantile");
                        }
                        
                        else if(fourSquareType.contains("Show Room")){
                                System.out.println("Commercial & Mercantile");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Commercial & Mercantile");
                        }
                        
                        
                        else if(fourSquareType.contains("Convenience Store")){
                                System.out.println("Commercial & Mercantile");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Commercial & Mercantile");
                        }

                        
                        else if(fourSquareType.contains("Hotel")){
                                System.out.println("Hotels & Restaurants");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hotels & Restaurants");
                        }
                        
                        else if(fourSquareType.contains("Restaurant")){
                                System.out.println("Hotels & Restaurants");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hotels & Restaurants");
                        }
                        
                        else if(fourSquareType.contains("Shop")){
                                System.out.println("Hotels & Restaurants");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hotels & Restaurants");
                        }
                        
                        else if(fourSquareType.contains("Pharmacy")){
                                System.out.println("Hotels & Restaurants");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hotels & Restaurants");
                        }
                        
                        else if(fourSquareType.matches("(?i).*Pub*")){
                                System.out.println("Hotels & Restaurants");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hotels & Restaurants");
                        }
                        
                        else if(fourSquareType.contains("Bar")){
                                System.out.println("Hotels & Restaurants");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hotels & Restaurants");
                        }
                        
                        else if(fourSquareType.contains("KFC")){
                                System.out.println("Hotels & Restaurants");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hotels & Restaurants");
                        }
                        
                        else if(fourSquareType.contains("Pizza Place")){
                                System.out.println("Hotels & Restaurants");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hotels & Restaurants");
                        }
                        
                        else if(fourSquareType.contains("Burger Joint")){
                                System.out.println("Hotels & Restaurants");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Hotels & Restaurants");
                        }
                        
                        else if(fourSquareType.contains("Office")){
                                System.out.println("Professional");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Professional");
                        }
                        
                        else if(fourSquareType.contains("Building")){
                                System.out.println("Professional");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Professional");
                        }
                        
                        else if(fourSquareType.contains("Embassy / Consulate")){
                                System.out.println("Professional");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Professional");
                        }
                        
                        else if(fourSquareType.contains("Service")){
                                System.out.println("Professional");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Professional");
                        }
                        
                        else if(fourSquareType.contains("Service")){
                                System.out.println("Professional");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Professional");
                        }
                        
                        else if(fourSquareType.contains("Workshop")){
                                System.out.println("Professional");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Professional");
                        }
                        
                        else if(fourSquareType.contains("Factory")){
                                System.out.println("Professional");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Professional");
                        }
                        
                        else if(fourSquareType.contains("Bakery")){
                                System.out.println("Professional");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Professional");
                        }
                        
                        else if(fourSquareType.contains("Alternative Healer")){
                                System.out.println("Professional");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Professional");
                        }
                        
                        else if(fourSquareType.contains("Residential Building (Apartment / Condo")){
                                System.out.println("Residential");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Residential");          
                        }
                        
                        else if(fourSquareType.contains("Temple")){
                                System.out.println("Residential");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Residential");          
                        }
                        
                        else if(fourSquareType.contains("Church")){
                                System.out.println("Residential");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Residential");          
                        }
                        
                        else if(fourSquareType.contains("Mosque")){
                                System.out.println("Residential");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue("Residential");          
                        }
                        else{
                            System.out.println(" ");
                                HSSFCell cell = row.createCell(8);				
				cell.setCellValue(" ");
                        }
      

		}
		
		FileOutputStream fileOut = new FileOutputStream(excelFileName);
		
		//write this workbook to an Outputstream.
		wb.write(fileOut);
		fileOut.flush();
		fileOut.close();
	}


	public static void main(String[] args) throws IOException {
		
		
		String [][] foursquareDataEntry = new String[3379][7];
                for(int i=0;i<3379;i++)
                    for(int j=0;j<7;j++)
                        foursquareDataEntry[i][j]=" ";
                
                foursquareDataEntry = readXLSFile();
               
                
                writeXLSFile(foursquareDataEntry);
		

	}

}
