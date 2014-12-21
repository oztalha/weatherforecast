package edu.gmu.fall2013.cs780proj;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Hashtable;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;
import java.util.logging.Logger;

class Util1 {	
	public static List<String> getTokens(String aLine, String delim) {
		List<String> ret = new ArrayList<String>();
		StringTokenizer st = new StringTokenizer(aLine, delim);
		while (st.hasMoreTokens()) 
			ret.add(st.nextToken());
		return ret; 
	}
}

/*
 *  GMU Fall 2013, CS 780 Course Project
 *  Pre and Post Processor
 *  
 *  Author: Venkat Tadakamalla
 */
public class CS780ProjKagglePreAndPost {	
	protected static String[] classKindSyn = new String[] {
		"s1",
		"bad worst sick worse",
		"s3",
		"good happy nice fun excellent great awesome positive",		
		"s5",
		"today now currently",
		"tomorrow",
		"w3",
		"yesterday",
		"cloud fog gloom smog smoke steam vapor veil billow dimness fogginess frost haze haziness murk nebula obscurity overcast thunderhead brume",
		"cold chilly freezing bitter bleak brisk chilled cool frigid frosty intense raw chill icebox sharp stinging algid benumbed biting blasting boreal brumal frore gelid glacial hawkish hiemal inclement nipping nippy numbed numbing",
		"dry moistureless arid bare barren dehydrated dusty parched stale baked bald depleted desert desiccant desiccated drained evaporated exhausted impoverished sapped sear shriveled anhydrous athirst dried-up droughty hard juiceless",
		"hot blazing boiling heated red scorching sizzling warm blistering broiling burning calescent decalescent febrile fevered feverish feverous fiery flaming igneous incandescent fire ovenlike parching piping recalescent roasting scalding searing smoking steaming summery sweltry thermogenic tropic",
		"humid damp dank muggy oppressive soggy steamy sticky stifling sultry sweltering wet clammy irriguous mucky sodden sweaty watery",
		"hurricane violent cyclone typhoon tropical",
		"cant",
		"ice chunk crystal diamonds floe glacier glaze hail hailstone iceberg icicle permafrost sleet",
		"other",
		"rain deluge drizzle flood mist precipitation rainfall rainstorm shower showers stream torrent condensation pour pouring raindrops sprinkle sprinkling volley",
		"snow blizzard snowfall",
		"storm squall",
		"sun sunlight bask daylight flare shine sol sunrise",
		"tornado twister whirlwind funnel",
		"wind breeze breath draft draught flutter mistral wafting whiff"};
	protected static List<TreeSet<String>> classKindSynList = null;
	
	protected static String[] classLabelNames = { "s1-can't tell", "s2-negative",
			"s3-neutral", "s4-positive", "s5-not related",
			"w1-current", "w2-future", "w3-can't tell",
			"w4-past weather", "k1-clouds", "k2-cold", "k3-dry",
			"k4-hot", "k5-humid", "k6-hurricane", "k7-can't tell",
			"k8-ice", "k9-other", "k10-rain", "k11-snow", "k12-storms",
			"k13-sun", "k14-tornado", "k15-wind" };
	protected static double[] classTotalCount = new double[classLabelNames.length];
	
	public static String JUNK_CHAR = " ,?;#.&:1234567890+-@(){}[]~!_\\/|$^%@*+='";
	protected static Logger logger = Logger.getLogger("edu.gmu.fall2013.cs780proj.Preprocessor.class");
	protected String dirPath = "O:/Eclipse_ws/1/CS780Proj/data/small/";
	protected String vocabFile = null;
	protected String vocabFileFull = null;
	protected String inpFile = null;	
	protected String inpFileFull = null;
	
    protected TreeSet<String> termsSet = null;
    protected List<String> termsList = null;
    protected List<String> lines = null;
    protected ArrayList<List<String>> linesToks = null;
    protected ArrayList<List<String>> linesLabels = null;
    protected TreeSet<String> stopWordSet = null;
    protected int testTrain = 0;
    
    public CS780ProjKagglePreAndPost(String vocabFile, String inpFile, int testTrain) {
    	this.vocabFile = vocabFile;
    	this.vocabFileFull = dirPath + vocabFile;
    	this.testTrain = testTrain;
    	
    	this.inpFile = inpFile;    	
    	this.inpFileFull = dirPath + inpFile;
    	//readInInpFile();
	}

	//method to completely stem the words in an array list
    public static ArrayList<String> completeStem(Collection<String> tokens1){
        PorterAlgo pa = new PorterAlgo();
        ArrayList<String> arrstr = new ArrayList<String>();
        for (String i : tokens1){
            String s1 = pa.step1(i);
            String s2 = pa.step2(s1);
            String s3= pa.step3(s2);
            String s4= pa.step4(s3);
            String s5= pa.step5(s4);
            arrstr.add(s5);
        }
        return arrstr;
    }   
   
	public void readInVocab() {		
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(vocabFileFull));
			termsSet = (TreeSet<String>) in.readObject();
			termsList= (List<String>) in.readObject();
			stopWordSet = (TreeSet<String>) in.readObject();
			classKindSynList = (List<TreeSet<String>>) in.readObject();
			System.out.println ("DEBUG: " + termsSet );
			System.out.println ("DEBUG: " + termsList );
			System.out.println ("DEBUG: " + stopWordSet );
			System.out.println ("DEBUG: " + classKindSynList);			
			in.close();		
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getStackTrace().toString());
		}
	}
	
	public void generateWekaFiles() {
		readInVocab();
		readInInpFile();
		int numLines = lines.size();
		int numTerms = termsList.size();
		System.out.println("numTerms: " + numTerms);
		System.out.println("numLines: " + numLines);
		accountForLabelSyn();
		boolean writePPFile = true;
		try {

			// xml file
			String inpFileFullXml = inpFileFull.replace(".csv", ".xml");			
			BufferedWriter bw = new BufferedWriter(new FileWriter(inpFileFullXml));			
			bw.write("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
			bw.write("<labels xmlns=\"http://mulan.sourceforge.net/labels\">\n");
			for (int i=1; i<=24; i++)
				bw.write("  <label name=\"L_" + i+ "\"></label>\n");
			bw.write("</labels>\n");
			bw.close();
			
			// arff file
			String inpFileFullPP = inpFileFull.replace(".csv", ".arff");			
			bw = new BufferedWriter(new FileWriter(inpFileFullPP));			
			bw.write("% CARPUS_VOCAB: [");
			for (int i=0; i<numTerms; i++)
				bw.write("  F_" + (i+25) + ":" + termsList.get(i));
			bw.write("  ]\n");			
						
			bw.write("@relation cloudsource_weather\n\n");
			bw.write("@attribute I_0 numeric\n");
			for (int i=1; i<=9; i++)
				bw.write("@attribute L_" + i + " {0, 1}\n");
			for (int i=10; i<=24; i++)
				bw.write("@attribute L_" + i + " {0, 1}\n");			
			for (int i=0; i<numTerms; i++)
				bw.write("@attribute F_" + (i+25) + " numeric\n");
			bw.write("\n@data\n");
			
			
			int printCounter = 0;
			classTotalCount = new double[24];
			for (int i = 0; i < numLines; i++) {
				List<String> toks = linesToks.get(i);
				int numToks = toks.size();
				int[] tf = new int[numTerms];
				for (int j = 0; j < numToks; j++) {
					String s1 = toks.get(j);
					for (int k = 0; k < numTerms; k++) {
						if (s1.equals(termsList.get(k))) {
							tf[k]++;
							break;
						}
					}
				}
				ArrayList<Integer> l1 = new ArrayList<Integer>();
				for (int j = 0; j < numTerms; j++) {
					if (tf[j] > 0) {
						l1.add(j);
						l1.add(tf[j]);
					}
				}

				if (writePPFile) {
					List<String> lineLabels = linesLabels.get(i);
					bw.write("{0 " + lineLabels.get(0));
					for (int j = 1; j <=24; j++) {
						classTotalCount[j-1] += new Double(lineLabels.get(j));
						bw.write(", " + j + " " + lineLabels.get(j));
					}
					for (int j = 0; j < l1.size(); j += 2) {
						int index = l1.get(j);
						int val = l1.get(j + 1);
						bw.write(", " + (index + 25) + " " + val);
					}
					bw.write("}\n");
				}			
				
				if (++printCounter == 5000) {
					System.out.println("Processed records: " + i);
					printCounter = 0;
				}
			}
			bw.close();
			for (int i=0; i<24; i++) {
				int cnt    = (int) classTotalCount[i];
				double percent= classTotalCount[i] * 100.0 / 77946.0;
				System.out.println ( classLabelNames[i] + ": " + cnt + ", " + percent);
			}
			System.out.println ("generateWekaFiles(): Success!");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void generateStatisticaFiles() {
		readInVocab();
		readInInpFile();
		accountForLabelSyn();
		int numLines = lines.size();
		int numTerms = termsList.size();
		System.out.println("numTerms: " + numTerms);
		System.out.println("numLines: " + numLines);

		try {			
			String inpFileFullPP = inpFileFull.replace(".csv", ".stat");			
			BufferedWriter bw = new BufferedWriter(new FileWriter(inpFileFullPP));	
			int printCounter = 0;
			bw.write("id,tstTrn,trnBin,s-1,s-2,s-3,s-4,s-5,w-1,w-2,w-3,w-4,k-1,k-2,k-3,k-4,k-5,k-6,k-7,k-8,k-9,k-10,k-11,k-12,k-13,k-14,k-15");
			for (int i=0; i<numTerms; i++) 
				bw.write(",v-" + (i+1));
			bw.write("\n");
			
			for (int i = 0; i < numLines; i++) {
				List<String> toks = linesToks.get(i);
				int numToks = toks.size();
				int[] tf = new int[numTerms];
				for (int j = 0; j < numToks; j++) {
					String s1 = toks.get(j);
					for (int k = 0; k < numTerms; k++) {
						if (s1.equals(termsList.get(k))) {
							tf[k]++;
							break;
						}
					}
				}
				ArrayList<Integer> l1 = new ArrayList<Integer>();
				for (int j = 0; j < numTerms; j++) {
					if (tf[j] > 0) {
						l1.add(j);
						l1.add(tf[j]);
					}
				}
				
				List<String> lineLabels = linesLabels.get(i);
				
				int trainBin = (testTrain!=0) ? 100 : (i/10000);
				bw.write(lineLabels.get(0) + "," + testTrain + "," + trainBin);
				for (int j = 1; j <=24; j++) {
					bw.write("," + lineLabels.get(j));
				}
				int counter = 0;
				for (int j = 0; j < l1.size(); j += 2) {
					int index = l1.get(j);
					while (counter<index) {
						bw.write(",0");	
						counter++;
					}
					int val = l1.get(j + 1);
					bw.write("," + val);
					counter++;
				}
				for (int j=counter; j<numTerms; j++)
					bw.write(",0");
				bw.write("\n");
				
				if (++printCounter == 5000) {
					System.out.println("Processed records: " + i);
					printCounter = 0;
				}
			}
			bw.close();
			System.out.println ("generateStatisticaFiles(): Success!");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}	

	public void readInInpFile() {
		try {
			BufferedReader br = new BufferedReader(new FileReader(inpFileFull));
			lines = new ArrayList<String>();
			linesLabels = new ArrayList<List<String>>();
			String aLine = br.readLine(); // skip the header
			while ((aLine = br.readLine()) != null) {
				List<String> tok = Util1.getTokens(aLine, "`");			
				lines.add(tok.get(1) + " " + tok.get(2) + " " + tok.get(3));
				List<String> lineLabels = new ArrayList<String>();
				lineLabels.add(tok.get(0));
				for (int i=4; i<=27; i++)
					lineLabels.add(tok.get(i));
				linesLabels.add(lineLabels);
			}
			buildLinesToks();
			br.close();		
			System.out.println ("readInInpFile(): Success!");
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getStackTrace().toString());
		}
	}
	
	public void buildLinesToks() {
		try {			
			linesToks = new ArrayList<List<String>>();
			for (int i = 0; i < lines.size(); i++) {
				String line = lines.get(i);
				if (null != line) {
					line = line.toLowerCase();
					line = line.replaceAll("[^a-zA-Z ]", "");
				}
				List<String> tok1 = Util1.getTokens(line, CS780ProjKagglePreAndPost.JUNK_CHAR);
				List<String> tok2 = completeStem(tok1);
				linesToks.add(tok2);
			}			
			
			System.out.println ("buildLinesToks(): Success!");
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getStackTrace().toString());
		}
	}
	
	
	
	private void generateCorpusVocab() {
		try {
			classKindSynList = new ArrayList<TreeSet<String>>();
			for (int i=0; i<classKindSyn.length; i++) {
				String kindItem = classKindSyn[i];
				List<String> tok1 = Util1.getTokens(kindItem, ", ");
				List<String> tok2 = completeStem(tok1);
				TreeSet<String> ts1 = new TreeSet<String>();
				for (String s1: tok2)  ts1.add(s1);
				classKindSynList.add(ts1);
			}			
			
			BufferedReader br = new BufferedReader(new FileReader("O:/Eclipse_ws/1/CS780Proj/data/all_stop_words_english.txt"));
			String aLine = null;
			stopWordSet = new TreeSet<String>();
			while ((aLine = br.readLine()) != null) 				
				stopWordSet.add(aLine.toLowerCase());
			br.close();
			
			termsSet = new TreeSet<String>();
			Hashtable<String, Integer> ht = new Hashtable<String, Integer>();
			readInInpFile();
			for (int i = 0; i < lines.size(); i++) {
				String line = lines.get(i);
				if (null != line) {
					line = line.toLowerCase();
					line = line.replaceAll("[^a-zA-Z ]", "");
				}
				List<String> tok1 = Util1.getTokens(line, CS780ProjKagglePreAndPost.JUNK_CHAR);
				List<String> tok2 = completeStem(tok1);
				for (String s1 : tok2) {
//					if (stopWordSet.contains(s1))
//						System.out.println ( "[DEBUG: Stop Word Found] " + s1);
					if (!termsSet.contains(s1)) {
						if (!stopWordSet.contains(s1)) {
							termsSet.add(s1);
							ht.put(s1, 1);
						}
					} else {
						Integer i1 = ht.get(s1);
						ht.remove(s1);
						ht.put(s1, i1 + 1);
					}
				}
			}
			Set<String> keys = ht.keySet();
			TreeSet<String> termsSet2 = new TreeSet<String>();
			for (String key : keys) {
				Integer val = ht.get(key);				
				// TODO: Restore the statement below
				if (null != val && (val > 50 || key.startsWith("label_")) && key.length() > 1)
				//if (null != val && key.length() > 1)
					termsSet2.add(key);
			}
			System.out.println("termsSet.size: " + termsSet.size() + ", " + termsSet);
			
			termsSet = termsSet2;
			// Add synonym related vocab...
			for (int i=1; i<25; i++) termsSet.add("label_"+i);			
			termsList = new ArrayList<String>();
			for (String s : termsSet) {
//				System.out.println(s);
				termsList.add(s);
			}
			System.out.println("reduced termsSet.size: " + termsSet.size() + ", " + termsSet);

			// Write it out so that we don't need regenerate everytime...
			String vocabFileFull = dirPath + vocabFile;
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(vocabFileFull)); 
			out.writeObject(termsSet);
			out.writeObject(termsList);
			out.writeObject(stopWordSet);
			out.writeObject(classKindSynList);
			out.close();
			
			String vocabFileFull_ = vocabFileFull.replace(".ser", ".csv");
			BufferedWriter bw = new BufferedWriter(new FileWriter(vocabFileFull_));
			for (int i=0; i<termsList.size(); i++)
				bw.write(i + " " + termsList.get(i) + "\n");
			bw.close();
			System.out.println("Generated Vocab File(s): " + vocabFileFull);

		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getStackTrace().toString());
		}
	}
	
	private void appendEmptyLabelsToTestRecords(String testFile) {
		try {			
			boolean flag = true;
			String append = "`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0`0";
			String testFileFull = dirPath + testFile;
			String testFileFull_ = dirPath + testFile.replace(".csv", "_.csv");
			BufferedReader br = new BufferedReader(new FileReader(testFileFull));
			BufferedWriter bw = new BufferedWriter(new FileWriter(testFileFull_));
			String aLine = null;
			while ((aLine = br.readLine()) != null) {
				if (null!=aLine && aLine.contains(append)) {
					flag = false;
					break;
				}
				bw.write(aLine + append +"\n");
			}
			br.close();
			bw.close();
			
			if (!flag) {				
				System.out.println ("ERROR: Already Appended...");
			}
			
			System.out.println ("oneTime(): exiting");
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getStackTrace().toString());
		}
	}
	public void combineTestTrainFiles(String r1, String r2, String w) {
		String r1Full = dirPath + r1;
		String r2Full = dirPath + r2;
		String wFull = dirPath + w;
		try {
			BufferedReader br1 = new BufferedReader(new FileReader(r1Full));
			BufferedReader br2 = new BufferedReader(new FileReader(r2Full));
			BufferedWriter bw = new BufferedWriter(new FileWriter(wFull));
			String aLine = null;
			while ((aLine = br1.readLine()) != null) {
				if (null!=aLine && aLine.length()>1) {
					bw.write(aLine + "\n");
				}
			}
			br1.close();
			aLine = br2.readLine(); // skip the header in the test file...
			while ((aLine = br2.readLine()) != null) {
				if (null!=aLine && aLine.length()>1) {
					bw.write(aLine + "\n");
				}
			}
			br2.close();
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getStackTrace().toString());
		}
	}
	
	public static double[] topKSpecial(double[] valsAll, int begin, int len, int k, boolean normalize, boolean returnOnes, double returnOnesThreshold) {
		double vals[] = new double[len];
		for (int i=0; i<vals.length; i++) vals[i] = valsAll[begin+i];
		
		double[] maxElems= new double[k];
		int[] maxIndices   = new int[k];
		
		double[] valsTemp = vals.clone();
//		System.out.println ( Arrays.toString(valsTemp) );
		for (int i = 0; i < k; i++) {
			maxElems[i] = Double.MIN_VALUE;
			maxIndices[i] = -1;
			for (int j = 0; j < len; j++) {
				if (maxElems[i] < valsTemp[j]) {
					maxElems[i] = valsTemp[j];
					maxIndices[i] = j;
				}
			}
			valsTemp[maxIndices[i]]=Double.MIN_VALUE; // Temporarily mess-up, okay temp structure.
		}	
		
//		System.out.println ( Arrays.toString(maxIndices) );
		valsTemp = vals.clone(); // Restore valsTemp as we messed-up temporarily
		
		// Zero-out smaller ones
		for (int i=0; i<len; i++) {
			boolean found = false;
			for (int j=0; j<k; j++) {
				if (i==maxIndices[j]) { 
					found = true;
					break;
				}
			}
			if (!found) valsTemp[i] = 0.0;
		}
	
		// Normalize the topK
		if (normalize) {
			double sum = 0.0;
			for (int i=0; i<k; i++) sum += valsTemp[maxIndices[i]];
			for (int i=0; i<k; i++) valsTemp[maxIndices[i]] /= sum;			
		}
		if (returnOnes) {
			for (int i=0; i<k; i++) {
				if (valsTemp[maxIndices[i]]>=returnOnesThreshold)
					valsTemp[maxIndices[i]] = 1.0;			
			}
			int count = 0;
			for (int i=0; i<k; i++) {
				if (valsTemp[maxIndices[i]] == 1.0) count++;
			}
			if (count<k) {
				double[] maxElems2 = new double[count];
				int[] maxIndices2 = new int[count];
				for (int i = 0; i < k; i++) {
					if (valsTemp[maxIndices[i]] == 1.0) {
						maxElems2[i] = 1.0;
						maxIndices2[i] = maxIndices[i];
					}
				}
				maxElems = maxElems2;
				maxIndices = maxIndices2;				
				for (int i=0; i<len; i++) {
					if (valsTemp[i]!=1.0) valsTemp[i] = 0.0;
				}
			}			
		}
		return valsTemp;
	}
	
	
	public void accountForLabelSyn() {
//		readInVocab();
//		readInInpFile();
		int numLines = lines.size();
		int numTerms = termsList.size();
		System.out.println("numTerms: " + numTerms);
		System.out.println("numLines: " + numLines);
		int gtpos =0, gpos=0, gfpos = 0;
		double gksum = 0.0;
		for (int i=0; i<lines.size(); i++) {
//		for (int i=0; i<50; i++) {
			String line = lines.get(i);
			// Test for Kind labels only
			int tpos=0;
			int fpos=0;
			int pos=0;
			double ksum = 0.0;
			List<String> ts1 = linesToks.get(i);
			//System.out.println ("\n" + ts1);
			List<String> labelI = linesLabels.get(i);
			for (int j=10; j<25; j++) {
				String labelJ = labelI.get(j);
				ksum += new Double(labelJ);
				if (!labelJ.equals("0")) {
					pos++;
					//System.out.println ("Label not zero " + j);
					TreeSet<String> ts2 = classKindSynList.get(j-1);
					//System.out.println (ts2);
					boolean found = false;
					for (int k=0; k<ts1.size(); k++) {
						String synWord = ts1.get(k);
						if (ts2.contains(synWord)) {
							//System.out.println ("tp, i, syn: " + i + ", " + synWord);
							found = true;
							ts1.add("label_" + j);
							break;
						}							
					}
					if (found) tpos++;
				}
				else {
					//System.out.println ("Label zero " + j);
					TreeSet<String> ts2 = classKindSynList.get(j-1);
					//System.out.println (ts2);
					boolean found = false;
					for (int k=0; k<ts1.size(); k++) {
						String synWord = ts1.get(k);
						if (ts2.contains(synWord)) {
							//System.out.println ("fp, i, syn: " + i + ", " + synWord);
							found = true;
							ts1.add("label_" + j);
							break;
						}							
					}
					if (found) fpos++;
					
				}
			}
			gpos+=pos; gtpos +=tpos;  gfpos += fpos;
			gksum += ksum;
			//System.out.println ( i + ": \t[tp:" + tpos + ", fp:" + fpos + "] / " + pos +"\t" + linesToks.get(i) + "\t" + linesLabels.get(i) + "\t ksum=" + ksum);
		}
		System.out.println ( "DEBUG: Added syn words...");
		System.out.println ( "\n\nGrand Totals: \t[gtp:" + gtpos + ", gfp:" + gfpos + "] / " + gpos + ", \t gksum ave: " + gksum/lines.size());
	}
	
	// Generate Statistica test and train files...
	public static void main_1() {		
		CS780ProjKagglePreAndPost reader = new CS780ProjKagglePreAndPost("carpus_vocab.ser", "train.csv", 0);
//		reader.appendEmptyLabelsToTestRecords("test.csv");
		reader.generateCorpusVocab();
//		reader.accountForLabelSyn();
//		reader.generateWekaFiles();
		reader.generateStatisticaFiles();
		reader = new CS780ProjKagglePreAndPost("carpus_vocab.ser", "test.csv", 1);
//		reader.generateWekaFiles();
		reader.generateStatisticaFiles();
		reader.combineTestTrainFiles("train.stat", "test.stat", "testTrain.csv");
	}
	
	// Stemming and stop words 
	public static void main_2() {
		try {
			BufferedReader br = new BufferedReader(new FileReader("O:/Eclipse_ws/1/CS780Proj/data/all_stop_words_english.txt"));
			String aLine = null;
			Set<String> lines = new TreeSet<String>();
			while ((aLine = br.readLine()) != null) {
				aLine = aLine.toLowerCase();
				if (!lines.contains((aLine)) && aLine.length()>0)
					lines.add(aLine);
			}
			System.out.println  (lines);
			List<String> temp2 = completeStem(lines);

			lines = new TreeSet<String>();
			for (String s: temp2)
				if (!lines.contains((s)))
					lines.add(s);
			br.close();
			System.out.println ( lines );
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getStackTrace().toString());
		}
	}
	
	// Normalize the results to upload to kaggle website...
	public static void main_3() {
		try {
			int numLables = 24;
			
			String path    = "O:/Eclipse_ws/1/CS780Proj/data/small/statistica/";
			String inpFile = path + "113013a_RandomForests1.csv";
			String outFile = inpFile.replace(".csv", "_mod.csv");
					
			BufferedReader br = new BufferedReader(new FileReader(inpFile));
			BufferedWriter bw = new BufferedWriter(new FileWriter(outFile));
			String aLine = br.readLine(); // Ignore the header
			bw.write("id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15\n");

			while ((aLine = br.readLine()) != null) {
				List<String> tok = Util1.getTokens(aLine, ",");
				double[] vals    = new double[numLables];
				String id = tok.get(0);
				for (int i=1; i<tok.size(); i++)
					vals[i-1] = new Double(tok.get(i));
				
				double senti[] = topKSpecial(vals, 0, 5, 2, true, false, 0.4);  // Sentiment
				double when[] = topKSpecial(vals, 5, 4, 2, true, false, 0.4);   // When		
				double kind[] = topKSpecial(vals, 9, 15, 4, false, false, 0.4); // Kind
				
				double[] valsMod = new double[numLables];
				int counter = 0;
				for (int i=0; i<senti.length; i++) valsMod[counter++]= senti[i];
				for (int i=0; i<when.length;  i++) valsMod[counter++]= when[i];
				for (int i=0; i<kind.length;  i++) valsMod[counter++]= kind[i];
				
				bw.write(id);
				for (int i=0; i<numLables; i++) {
					classTotalCount[i] += valsMod[i];
					bw.write("," + valsMod[i]);
				}
				bw.write("\n");
			}
			for (int i=0; i<24; i++) {
				int cnt    = (int) classTotalCount[i];
				double percent= classTotalCount[i] / 42157.0;
				System.out.println ( classLabelNames[i] + ": " + cnt + ", " + percent);
			}
			br.close();
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getStackTrace().toString());
		}
	}	
	
	public static void main(String[] args) {
		// main_1();
		// main_2();
		main_3();
	}	
}
