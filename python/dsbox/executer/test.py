def _create_pipeline_executable(self, pipeline, pipeid):
    imports = ["sklearn.externals", "pandas"]
        statements = ["", "# Pipeline : %s" % str(pipeline), ""]
        statements.append("testdata = pandas.read_csv('%s/data/testData.csv.gz', index_col='%s')" % (self.directory, self.indexcol))
        index = 1
        for primitive in pipeline:
            primid = "primitive_%s" % str(index)
            mod, cls = primitive.cls.rsplit('.', 1)
            imports.append(mod)
            primfile = "%s/models/%s.%s.pkl" % (self.outputdir, pipeid, primid)
            
            # Initialize primitive
            if primitive.is_persistent:
                statements.append("%s = sklearn.externals.joblib.load('%s')" % (primid, primfile))
                joblib.dump(primitive.executable, primfile)
            else:
                args = []
                kwargs = {}
                if primitive.getInitKeywordArgs():
                    kwargs = self.l2_planner._process_kwargs(primitive.getInitKwargs(), self.task_type, self.metric)
                if primitive.getInitArgs():
                    args = self.l2_planner._process_args(primitive.getInitArgs(), self.task_type, self.metric)
                statements.append("args = %s" % args)
                statements.append("kwargs = %s" % kwargs)
                statements.append("%s = %s(*args, **kwargs)" % (primid, primitive.cls))
            
            if primitive.task == "Modeling":
                statements.append("print %s.predict(testdata)" % primid)
            else:
                if primitive.task == "PreProcessing":
                    if primitive.column_primitive:
                        statements.append("for col in testdata.columns:")
                        statements.append("    testdata[col] = %s.fit_transform(testdata[col])" % primid)
                    else:
                        statements.append("testdata = %s.fit_transform(testdata)" % primid)
            
                elif primitive.task == "FeatureExtraction":
                    if primitive.column_primitive:
                        statements.append("for col in testdata.columns:")
                        statements.append("    testdata[col] = %s.fit_transform(testdata[col])" % primid)
                    else:
                        statements.append("testdata = %s.fit_transform(testdata)" % primid)

            index += 1
    
    with open("%s/executables/%s.py" % (self.outputdir, pipeid), 'a') as exfile:
        for imp in set(imports):
            exfile.write("import %s\n" % imp)
            for st in statements:
                exfile.write(st+"\n")

def _load_json(self, jsonfile):
    with open(jsonfile) as json_data:
        d = json.load(json_data)
            json_data.close()
            return d
