import 'package:flutter/material.dart';
import 'package:legal_text_app/utils/styles.dart';

class AboutScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.midnightBlack,
      appBar: AppBar(
        title: Text('About Us'),
        backgroundColor: AppColors.darkBlue,
      ),
      body: Padding(
        padding: EdgeInsets.all(20),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Image.asset('assets/icon.png', height: 100),
              ),
              SizedBox(height: 20),
              Text('Legal Text Processor', style: AppTextStyles.titleLarge),
              SizedBox(height: 20),
              Text(
                'Our app simplifies legal document handling with:',
                style: AppTextStyles.bodyMedium,
              ),
              SizedBox(height: 10),
              _buildFeatureItem(Icons.summarize, 'Document Summarization'),
              _buildFeatureItem(Icons.visibility_off, 'Content Anonymization'),
              _buildFeatureItem(Icons.security, 'Secure Processing'),
              SizedBox(height: 30),
              Text(
                'All documents are processed securely and never stored permanently.',
                style: TextStyle(color: AppColors.coolGray),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFeatureItem(IconData icon, String text) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(icon, color: AppColors.electricCyan),
          SizedBox(width: 10),
          Text(text, style: AppTextStyles.bodyMedium),
        ],
      ),
    );
  }
}
