import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:legal_text_app/utils/styles.dart';
import 'dart:io';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool isProcessing = false;
  bool summarize = true;
  bool anonymize = true;
  String? summary;
  dynamic processedFile;
  String? fileName;
  PlatformFile? selectedFile;

  Future<void> pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['docx', 'pdf', 'txt'],
    );
    if (result != null) {
      setState(() {
        selectedFile = result.files.first;
        fileName = selectedFile!.name;
      });
    }
  }

  Future<void> processFile() async {
    if (selectedFile == null) return;
    setState(() => isProcessing = true);

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://localhost:8000/process'),
      );
      request.files.add(http.MultipartFile.fromBytes(
          'file', selectedFile!.bytes!,
          filename: fileName));
      request.fields['summarize'] = summarize.toString();
      request.fields['anonymize'] = anonymize.toString();

      var response = await request.send();
      var responseData = await http.Response.fromStream(response);

      if (response.statusCode == 200) {
        var data = jsonDecode(responseData.body);
        setState(() {
          summary = data['summary'];
          processedFile = base64Decode(data['processed_file']);
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Processing failed: ${response.statusCode}'),
            backgroundColor: AppColors.neonGreen,
          ),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e'),
          backgroundColor: AppColors.neonGreen,
        ),
      );
    } finally {
      setState(() => isProcessing = false);
    }
  }

  Future<void> downloadFile() async {
    if (processedFile == null) return;

    final directory = await getTemporaryDirectory();
    final file = File('${directory.path}/$fileName');
    await file.writeAsBytes(processedFile);

    // Open the file in browser preview
    await launchUrl(Uri.file(file.path));

    // For mobile, show a save dialog
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('File saved to ${file.path}'),
        backgroundColor: AppColors.neonGreen,
      ),
    );
  }

   void _showAboutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.darkBlue,
        title: Text('About Legal Text Processor',
            style: TextStyle(color: AppColors.white)),
        content: SingleChildScrollView(
          child: Text(
            'This app helps you process legal documents by:\n\n'
            '• Summarizing lengthy legal texts\n'
            '• Anonymizing sensitive information\n\n'
            'Simply upload your document (PDF, DOCX, or TXT), '
            'select your processing options, and get your results instantly.',
            style: AppTextStyles.bodyMedium,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child:
                Text('Close', style: TextStyle(color: AppColors.electricCyan)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.midnightBlack,
      appBar: AppBar(
        backgroundColor: AppColors.darkBlue,
        leading: IconButton(
          icon: Icon(Icons.home, color: AppColors.white),
          onPressed: () {
            setState(() {
              selectedFile = null;
              summary = null;
              isProcessing = false;
            });
          },
        ),
        title: Text('Legal Text Processor', style: AppTextStyles.titleLarge),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(Icons.info_outline, color: AppColors.white),
            onPressed: () => _showAboutDialog(context),
          ),
        ],
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: BoxConstraints(maxWidth: 500), // Limits maximum width
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Upload and Processing Box
                Container(
                  padding: EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: AppColors.darkBlue,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // File Upload Section
                      ElevatedButton(
                        onPressed: pickFile,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primaryBlue,
                          foregroundColor: AppColors.white,
                          minimumSize: Size(200, 50),
                        ),
                        child: Text(
                          selectedFile == null 
                            ? 'Upload Document' 
                            : 'Change Document',
                          style: TextStyle(fontSize: 16),
                        ),
                      ),
                      SizedBox(height: 16),
                      if (selectedFile != null) 
                        Text(
                          'Selected: $fileName',
                          style: AppTextStyles.bodyMedium,
                          textAlign: TextAlign.center,
                        ),
                      SizedBox(height: 24),
                      
                      // Processing Options
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Checkbox(
                            value: summarize,
                            onChanged: (val) => setState(() => summarize = val!),
                            activeColor: AppColors.primaryBlue,
                          ),
                          Text('Summarize', style: AppTextStyles.bodyMedium),
                          SizedBox(width: 20),
                          Checkbox(
                            value: anonymize,
                            onChanged: (val) => setState(() => anonymize = val!),
                            activeColor: AppColors.primaryBlue,
                          ),
                          Text('Anonymize', style: AppTextStyles.bodyMedium),
                        ],
                      ),
                      SizedBox(height: 24),
                      
                      // Process Button
                      if (selectedFile != null)
                        ElevatedButton(
                          onPressed: isProcessing ? null : processFile,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: isProcessing 
                              ? AppColors.coolGray 
                              : AppColors.primaryBlue,
                            foregroundColor: AppColors.white,
                            minimumSize: Size(200, 50),
                          ),
                          child: isProcessing
                            ? CircularProgressIndicator(color: AppColors.white)
                            : Text('Process Document', style: TextStyle(fontSize: 16)),
                        ),
                    ],
                  ),
                ),
                SizedBox(height: 32),
                
                // Results Section
                if (summary != null) 
                  Container(
                    padding: EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: AppColors.darkBlue,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Results', style: AppTextStyles.titleLarge),
                        SizedBox(height: 16),
                        
                        if (summarize) ...[
                          Text('Summary:', style: TextStyle(
                            color: AppColors.white,
                            fontWeight: FontWeight.bold,
                          )),
                          SizedBox(height: 8),
                          Container(
                            padding: EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: AppColors.midnightBlack,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: SelectableText(
                              summary!,
                              style: AppTextStyles.bodyMedium,
                            ),
                          ),
                          SizedBox(height: 16),
                        ],
                        
                        // Download Buttons
                        Center(
                          child: Wrap(
                            spacing: 20,
                            runSpacing: 20,
                            children: [
                              if (summarize)
                                ElevatedButton(
                                  onPressed: () {
                                    // Export summary functionality
                                  },
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: AppColors.purpleAccent,
                                    foregroundColor: AppColors.white,
                                  ),
                                  child: Text('Export Summary'),
                                ),
                              
                              if (anonymize)
                                ElevatedButton(
                                  onPressed: downloadFile,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: AppColors.primaryBlue,
                                    foregroundColor: AppColors.white,
                                  ),
                                  child: Text('Download Anonymized'),
                                ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}