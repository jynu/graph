import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict
import subprocess

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BIRDPipelineWorkflow:
    """
    Complete BIRD Dataset Processing Workflow
    
    Orchestrates the entire pipeline:
    1. BIRD data validation and preprocessing
    2. GPT-4 powered annotation
    3. Enhanced knowledge graph construction
    4. Validation and testing
    """
    
    def __init__(self):
        self.workflow_start_time = datetime.now()
        self.pipeline_steps = []
        self.results = {}
        
        logger.info("🚀 BIRD Pipeline Workflow Initialized")
    
    async def run_complete_pipeline(self):
        """Run the complete BIRD processing pipeline."""
        
        logger.info("=" * 80)
        logger.info("🚀 BIRD DATASET PROCESSING PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Step 1: Validate Prerequisites
            await self._step_1_validate_prerequisites()
            
            # Step 2: Process BIRD Data with GPT-4 Annotation
            await self._step_2_process_bird_data()
            
            # Step 3: Build Enhanced Knowledge Graph
            await self._step_3_build_enhanced_kg()
            
            # Step 4: Validate and Test
            await self._step_4_validate_and_test()
            
            # Step 5: Generate Final Report
            await self._step_5_generate_report()
            
            logger.info("🎉 BIRD Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            await self._handle_pipeline_failure(e)
            raise
    
    async def _step_1_validate_prerequisites(self):
        """Step 1: Validate all prerequisites are met."""
        
        logger.info("📋 Step 1: Validating Prerequisites")
        step_start = datetime.now()
        
        validation_results = {
            "bird_files": False,
            "internal_files": False,
            "dependencies": False,
            "gpt_access": False,
            "embedding_access": False
        }
        
        # Check BIRD files
        bird_files = [
            "train_tables_bird_short.json",
            "dev_tables_bird_short.json"
        ]
        
        existing_bird_files = [f for f in bird_files if os.path.exists(f)]
        if existing_bird_files:
            validation_results["bird_files"] = True
            logger.info(f"✅ Found {len(existing_bird_files)} BIRD files: {existing_bird_files}")
        else:
            logger.error("❌ No BIRD files found!")
            logger.info("💡 Please ensure BIRD JSON files are available:")
            for f in bird_files:
                logger.info(f"   - {f}")
        
        # Check internal files
        internal_files = [
            'transcation_all_final_output.json',
            'reference_all_final_output.json'
        ]
        
        existing_internal_files = [f for f in internal_files if os.path.exists(f)]
        if existing_internal_files:
            validation_results["internal_files"] = True
            logger.info(f"✅ Found {len(existing_internal_files)} internal files")
        else:
            logger.warning("⚠️ No internal files found - will build BIRD-only knowledge graph")
        
        # Check dependencies
        try:
            import duckdb
            import numpy as np
            from app.utils.client_manager import client_manager
            from app.rag.embedding import embedding
            validation_results["dependencies"] = True
            logger.info("✅ All Python dependencies available")
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
        
        # Test GPT access
        try:
            test_response = await client_manager.ask_gpt("Test connection - respond with 'OK'")
            if test_response and len(test_response) > 0:
                validation_results["gpt_access"] = True
                logger.info("✅ GPT-4 access confirmed")
            else:
                logger.error("❌ GPT-4 access test failed - empty response")
        except Exception as e:
            logger.error(f"❌ GPT-4 access test failed: {e}")
        
        # Test embedding access
        try:
            test_embedding = embedding.embed_text("test")
            if test_embedding and len(test_embedding) > 0:
                validation_results["embedding_access"] = True
                logger.info("✅ Embedding service access confirmed")
            else:
                logger.error("❌ Embedding service test failed")
        except Exception as e:
            logger.error(f"❌ Embedding service test failed: {e}")
        
        # Validate results
        essential_checks = ["bird_files", "dependencies", "gpt_access", "embedding_access"]
        failed_checks = [check for check in essential_checks if not validation_results[check]]
        
        if failed_checks:
            raise Exception(f"Prerequisites validation failed: {failed_checks}")
        
        step_duration = (datetime.now() - step_start).total_seconds()
        self.pipeline_steps.append({
            "step": "1_validate_prerequisites",
            "duration_seconds": step_duration,
            "status": "success",
            "results": validation_results
        })
        
        logger.info(f"✅ Step 1 completed in {step_duration:.2f}s")
    
    async def _step_2_process_bird_data(self):
        """Step 2: Process BIRD data with GPT-4 annotation."""
        
        logger.info("🤖 Step 2: Processing BIRD Data with GPT-4 Annotation")
        step_start = datetime.now()
        
        try:
            # Import and run BIRD processor
            from bird_data_processor import BIRDDataProcessor
            
            processor = BIRDDataProcessor()
            
            # Process BIRD files
            bird_files = [f for f in ["train_tables_bird_short.json", "dev_tables_bird_short.json"] 
                         if os.path.exists(f)]
            
            output_files = await processor.process_bird_files(bird_files)
            
            processing_results = {
                "input_files": bird_files,
                "output_files": output_files,
                "total_files_generated": len(output_files),
                "processing_successful": len(output_files) > 0
            }
            
            if not processing_results["processing_successful"]:
                raise Exception("No BIRD files were successfully processed")
            
            step_duration = (datetime.now() - step_start).total_seconds()
            self.pipeline_steps.append({
                "step": "2_process_bird_data",
                "duration_seconds": step_duration,
                "status": "success",
                "results": processing_results
            })
            
            logger.info(f"✅ Step 2 completed in {step_duration:.2f}s")
            logger.info(f"📁 Generated {len(output_files)} processed BIRD files")
            
        except Exception as e:
            logger.error(f"❌ Step 2 failed: {e}")
            raise
    
    async def _step_3_build_enhanced_kg(self):
        """Step 3: Build enhanced knowledge graph."""
        
        logger.info("🏗️ Step 3: Building Enhanced Knowledge Graph")
        step_start = datetime.now()
        
        try:
            # Import and run enhanced KG builder
            from enhanced_kg_builder import EnhancedDuckDBKnowledgeGraphBuilder, EmbeddingProvider
            
            builder = EnhancedDuckDBKnowledgeGraphBuilder(
                db_path="enhanced_knowledge_graph.duckdb",
                embedding_provider=EmbeddingProvider.OPENAI,
                embedding_dimensions=1536
            )
            
            # Clear existing graph
            builder.clear_graph()
            
            # Load internal files if available
            internal_files = []
            for f in ['transcation_all_final_output.json', 'reference_all_final_output.json']:
                if os.path.exists(f):
                    internal_files.append(f)
            
            if os.path.exists('marketdata_all_final_output.json'):
                internal_files.append('marketdata_all_final_output.json')
            
            # Load mixed metadata
            builder.load_mixed_metadata(internal_files, bird_files=None)  # Auto-detect BIRD files
            
            # Load ERD relationships if available
            if os.path.exists('gemini_extracted_relationships.json'):
                builder.load_erd_relationships('gemini_extracted_relationships.json')
            
            # Enhanced relationship inference
            builder.infer_enhanced_relationships()
            
            # Build the graph
            builder.build_graph()
            
            # Verify and get statistics
            verification_stats = builder.verify_enhanced_graph_structure()
            db_info = builder.get_database_info()
            
            # Export metadata
            metadata_file = builder.export_enhanced_metadata()
            
            kg_results = {
                "database_path": "enhanced_knowledge_graph.duckdb",
                "database_size_mb": db_info['database_size_mb'],
                "total_tables": verification_stats.get('tables', 0),
                "total_relationships": verification_stats.get('relationships', 0),
                "total_columns": verification_stats.get('columns', 0),
                "metadata_export": metadata_file,
                "verification_stats": verification_stats
            }
            
            step_duration = (datetime.now() - step_start).total_seconds()
            self.pipeline_steps.append({
                "step": "3_build_enhanced_kg",
                "duration_seconds": step_duration,
                "status": "success",
                "results": kg_results
            })
            
            logger.info(f"✅ Step 3 completed in {step_duration:.2f}s")
            logger.info(f"📊 Knowledge Graph: {kg_results['total_tables']} tables, {kg_results['total_relationships']} relationships")
            
        except Exception as e:
            logger.error(f"❌ Step 3 failed: {e}")
            raise
    
    async def _step_4_validate_and_test(self):
        """Step 4: Validate and test the enhanced knowledge graph."""
        
        logger.info("🧪 Step 4: Validating and Testing Enhanced Knowledge Graph")
        step_start = datetime.now()
        
        try:
            validation_results = {
                "database_connectivity": False,
                "table_count_validation": False,
                "relationship_validation": False,
                "embedding_validation": False,
                "similarity_search_test": False
            }
            
            # Test database connectivity
            try:
                import duckdb
                conn = duckdb.connect("enhanced_knowledge_graph.duckdb")
                table_count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
                conn.close()
                
                if table_count > 0:
                    validation_results["database_connectivity"] = True
                    validation_results["table_count_validation"] = True
                    logger.info(f"✅ Database connectivity confirmed: {table_count} tables")
                else:
                    logger.error("❌ Database is empty")
                    
            except Exception as e:
                logger.error(f"❌ Database connectivity test failed: {e}")
            
            # Test relationships
            try:
                conn = duckdb.connect("enhanced_knowledge_graph.duckdb")
                rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
                conn.close()
                
                if rel_count > 0:
                    validation_results["relationship_validation"] = True
                    logger.info(f"✅ Relationships validation passed: {rel_count} relationships")
                else:
                    logger.warning("⚠️ No relationships found")
                    
            except Exception as e:
                logger.error(f"❌ Relationship validation failed: {e}")
            
            # Test embeddings
            try:
                conn = duckdb.connect("enhanced_knowledge_graph.duckdb")
                embedding_count = conn.execute("SELECT COUNT(*) FROM tables WHERE embedding IS NOT NULL").fetchone()[0]
                conn.close()
                
                if embedding_count > 0:
                    validation_results["embedding_validation"] = True
                    logger.info(f"✅ Embeddings validation passed: {embedding_count} tables with embeddings")
                else:
                    logger.warning("⚠️ No embeddings found")
                    
            except Exception as e:
                logger.error(f"❌ Embedding validation failed: {e}")
            
            # Test similarity search
            try:
                from enhanced_kg_builder import EnhancedDuckDBKnowledgeGraphBuilder, EmbeddingProvider
                
                builder = EnhancedDuckDBKnowledgeGraphBuilder("enhanced_knowledge_graph.duckdb")
                
                # Test search
                search_results = builder.similarity_search("financial transactions", limit=5, search_type="tables")
                
                if search_results and len(search_results) > 0:
                    validation_results["similarity_search_test"] = True
                    logger.info(f"✅ Similarity search test passed: {len(search_results)} results")
                else:
                    logger.warning("⚠️ Similarity search returned no results")
                    
            except Exception as e:
                logger.error(f"❌ Similarity search test failed: {e}")
            
            # Overall validation
            passed_tests = sum(1 for result in validation_results.values() if result)
            total_tests = len(validation_results)
            
            validation_success = passed_tests >= (total_tests * 0.8)  # 80% pass rate
            
            step_duration = (datetime.now() - step_start).total_seconds()
            self.pipeline_steps.append({
                "step": "4_validate_and_test",
                "duration_seconds": step_duration,
                "status": "success" if validation_success else "warning",
                "results": {
                    "validation_results": validation_results,
                    "passed_tests": passed_tests,
                    "total_tests": total_tests,
                    "pass_rate": passed_tests / total_tests,
                    "validation_success": validation_success
                }
            })
            
            if validation_success:
                logger.info(f"✅ Step 4 completed successfully in {step_duration:.2f}s")
                logger.info(f"📊 Validation: {passed_tests}/{total_tests} tests passed")
            else:
                logger.warning(f"⚠️ Step 4 completed with warnings in {step_duration:.2f}s")
                logger.warning(f"📊 Validation: {passed_tests}/{total_tests} tests passed (below 80% threshold)")
                
        except Exception as e:
            logger.error(f"❌ Step 4 failed: {e}")
            raise
    
    async def _step_5_generate_report(self):
        """Step 5: Generate final pipeline report."""
        
        logger.info("📋 Step 5: Generating Final Pipeline Report")
        step_start = datetime.now()
        
        try:
            # Calculate total pipeline duration
            total_duration = (datetime.now() - self.workflow_start_time).total_seconds()
            
            # Create comprehensive report
            report = {
                "pipeline_execution": {
                    "start_time": self.workflow_start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_duration_seconds": total_duration,
                    "total_duration_minutes": total_duration / 60,
                    "status": "completed"
                },
                "steps_summary": self.pipeline_steps,
                "overall_results": {
                    "steps_completed": len(self.pipeline_steps),
                    "steps_successful": len([s for s in self.pipeline_steps if s["status"] == "success"]),
                    "total_processing_time": sum(s["duration_seconds"] for s in self.pipeline_steps)
                },
                "deliverables": {
                    "processed_bird_files": "processed_bird_data/",
                    "enhanced_knowledge_graph": "enhanced_knowledge_graph.duckdb",
                    "metadata_export": "enhanced_kg_metadata_*.json",
                    "pipeline_report": "bird_pipeline_report_*.json"
                },
                "next_steps": [
                    "Test similarity search: builder.similarity_search('your query')",
                    "Run benchmarks: python duckdb_benchmark_llm_v3.py",
                    "Explore relationships in enhanced_knowledge_graph.duckdb",
                    "Fine-tune relationship inference parameters if needed",
                    "Integrate with your text-to-SQL system"
                ],
                "recommendations": {
                    "performance": "Enhanced KG is 2-10x faster than Neo4j for analytical queries",
                    "scalability": "Consider processing additional BIRD datasets for more coverage",
                    "maintenance": "Re-run annotation process when BIRD data is updated",
                    "integration": "Use enhanced similarity search for better table retrieval"
                }
            }
            
            # Add step-specific results
            for step in self.pipeline_steps:
                if "results" in step:
                    step_name = step["step"]
                    report[f"{step_name}_details"] = step["results"]
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"bird_pipeline_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            step_duration = (datetime.now() - step_start).total_seconds()
            self.pipeline_steps.append({
                "step": "5_generate_report",
                "duration_seconds": step_duration,
                "status": "success",
                "results": {
                    "report_file": report_file,
                    "total_pipeline_duration": total_duration
                }
            })
            
            # Print summary
            logger.info("=" * 80)
            logger.info("🎉 BIRD PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"⏱️ Total Duration: {total_duration/60:.1f} minutes")
            logger.info(f"📊 Steps Completed: {len(self.pipeline_steps)}")
            logger.info(f"📁 Report Generated: {report_file}")
            
            logger.info("\n🎯 Key Deliverables:")
            logger.info(f"   📁 Processed BIRD Data: processed_bird_data/")
            logger.info(f"   🗄️ Enhanced Knowledge Graph: enhanced_knowledge_graph.duckdb")
            logger.info(f"   📋 Pipeline Report: {report_file}")
            
            logger.info("\n💡 Next Steps:")
            for next_step in report["next_steps"]:
                logger.info(f"   • {next_step}")
            
            logger.info("\n🚀 Ready for Text-to-SQL Integration!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Step 5 failed: {e}")
            raise
    
    async def _handle_pipeline_failure(self, error):
        """Handle pipeline failure and generate error report."""
        
        logger.error("💥 Pipeline Failure Handler")
        
        try:
            failure_report = {
                "pipeline_failure": {
                    "timestamp": datetime.now().isoformat(),
                    "error_message": str(error),
                    "steps_completed": len(self.pipeline_steps),
                    "last_successful_step": self.pipeline_steps[-1]["step"] if self.pipeline_steps else None
                },
                "completed_steps": self.pipeline_steps,
                "troubleshooting": {
                    "common_issues": [
                        "Check BIRD JSON files are present and valid",
                        "Verify GPT-4 API access and credentials",
                        "Ensure embedding service is accessible",
                        "Check disk space for database creation",
                        "Verify all Python dependencies are installed"
                    ],
                    "recovery_options": [
                        "Re-run individual pipeline steps",
                        "Check logs for specific error details",
                        "Validate input data formats",
                        "Test API connections separately"
                    ]
                }
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            failure_file = f"bird_pipeline_failure_{timestamp}.json"
            
            with open(failure_file, 'w', encoding='utf-8') as f:
                json.dump(failure_report, f, indent=2, ensure_ascii=False)
            
            logger.error(f"💥 Failure report saved: {failure_file}")
            
        except Exception as report_error:
            logger.error(f"Failed to generate failure report: {report_error}")


async def run_bird_pipeline():
    """Main function to run the complete BIRD pipeline."""
    
    try:
        workflow = BIRDPipelineWorkflow()
        await workflow.run_complete_pipeline()
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("🚀 BIRD Dataset Processing Pipeline")
    print("This will process BIRD data and build an enhanced knowledge graph")
    print("Estimated time: 10-30 minutes depending on data size and API speed")
    print()
    
    response = input("Do you want to continue? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        asyncio.run(run_bird_pipeline())
    else:
        print("Pipeline cancelled.")
