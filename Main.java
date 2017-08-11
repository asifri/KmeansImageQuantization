package HomeWork7;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;

import javax.imageio.ImageIO;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Main{
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
	
	private static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static Instances convertImgToInstances(BufferedImage image) {
		Attribute attribute1 = new Attribute("alpha");
		Attribute attribute2 = new Attribute("red");
		Attribute attribute3 = new Attribute("green");
		Attribute attribute4 = new Attribute("blue");
		ArrayList<Attribute> attributes = new ArrayList<Attribute>(4);
		attributes.add(attribute1);
		attributes.add(attribute2);
		attributes.add(attribute3);
		attributes.add(attribute4);
		Instances imageInstances = new Instances("Image", attributes, image.getHeight() * image.getWidth());

		int[][] result = new int[image.getHeight()][image.getWidth()];
		int[][][] resultARGB = new int[image.getHeight()][image.getWidth()][4];

		for (int col = 0; col < image.getWidth(); col++) {
			for (int row = 0; row < image.getHeight(); row++) {
				int pixel = image.getRGB(col, row);

				int alpha = (pixel >> 24) & 0xff;
				int red = (pixel >> 16) & 0xff;
				int green = (pixel >> 8) & 0xff;
				int blue = (pixel) & 0xff;
				result[row][col] = pixel;
				resultARGB[row][col][0] = alpha;
				resultARGB[row][col][1] = red;
				resultARGB[row][col][2] = green;
				resultARGB[row][col][3] = blue;

				Instance iExample = new DenseInstance(4);
				iExample.setValue((Attribute) attributes.get(0), alpha);// alpha
				iExample.setValue((Attribute) attributes.get(1), red);// red
				iExample.setValue((Attribute) attributes.get(2), green);// green
				iExample.setValue((Attribute) attributes.get(3), blue);// blue
				imageInstances.add(iExample);
			}
		}

		return imageInstances;

	}


	public static BufferedImage convertInstancesToImg(Instances instancesImage, int width, int height) {
		final BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		int index = 0;
		for (int col = 0; col < width; ++col) {
			for (int row = 0; row < height; ++row) {
				Instance instancePixel = instancesImage.instance(index);
				int pixel = ((int) instancePixel.value(0) << 24) | (int) instancePixel.value(1) << 16
						| (int) instancePixel.value(2) << 8 | (int) instancePixel.value(3);
				image.setRGB(col, row, pixel);
				index++;
			}
		}
		return image;
	}
	
	private static double calcAvgDistance(Instances originalSet, Instances transformedSet){
		double distance = 0;
		double numOfInstances = originalSet.numInstances();
		
		for (int i = 0; i < numOfInstances; i++) {
			distance += calcEuclideanDistance(originalSet.instance(i), transformedSet.instance(i));
		}
		
		return distance == 0 ? 0 : distance / numOfInstances;	
	}
	
	private static double calcEuclideanDistance(Instance x1, Instance x2){
    	double sumOfDistances = 0;
    	int dimensionOfVector = x1.numAttributes() - 1;
    	
    	for (int i = 0; i < dimensionOfVector; i++){
    		double attributeValueOfX1 = x1.value(i);
    		double attributeValueOfX2 = x2.value(i);
			sumOfDistances += Math.pow((attributeValueOfX1 - attributeValueOfX2), 2);
		}

    	return (sumOfDistances == 0) ? 0 : Math.pow(sumOfDistances, 0.5);
	}
	
	public static void KMeansSection() throws Exception{
		BufferedImage image = ImageIO.read(new File("messi.jpg"));
		int imageWidth = image.getWidth(); 
		int imageHeight = image.getHeight();
		Instances dataSet = convertImgToInstances(image);
		Instances quantizedInstances;
		int[] kValues = {2, 3, 5, 10, 25, 50, 100, 256};
		
		for (int i : kValues) {
			KMeans kMeans = new KMeans(i);
			Instances tempInstances = new Instances(dataSet);
			kMeans.buildClusterModel(tempInstances); 
			quantizedInstances = kMeans.quantize(tempInstances);
			BufferedImage quantizedImage = convertInstancesToImg(quantizedInstances, imageWidth, imageHeight);
			File outputfile = new File("output" + i +".jpg");
			ImageIO.write(quantizedImage, "jpg", outputfile);
		}
	}

	public static void main(String[] args) throws Exception {
		KMeansSection();
	}
}

