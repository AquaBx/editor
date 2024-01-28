

// These directories are all relative to the tests/ directory.


/// Arguments that modify test behaviour.
///
/// Specify them like this when developing:
/// `cargo test --workspace --test tests -- --help`
#[derive(Debug, Clone, Parser)]
#[clap(name = "typst-test", author)]
struct Args {
    /// All the tests that contains a filter string will be run (unless
    /// `--exact` is specified, which is even stricter).
    filter: Vec<String>,
    /// Runs only the specified subtest.
    #[arg(short, long)]
    #[arg(allow_hyphen_values = true)]
    subtest: Option<isize>,
    /// Runs only the test with the exact name specified in your command.
    ///
    /// Example:
    /// `cargo test --workspace --test tests  -- compiler/bytes.typ --exact`
    #[arg(long)]
    exact: bool,
    /// Updates the reference images in `tests/ref`.
    #[arg(long, default_value_t = env::var_os("UPDATE_EXPECT").is_some())]
    update: bool,
    /// Exports the tests as PDF into `tests/pdf`.
    #[arg(long)]
    pdf: bool,
    /// Configuration of what to print.
    #[command(flatten)]
    print: PrintConfig,
    /// Running `cargo test --workspace -- --nocapture` for the unit tests would
    /// fail the test runner without argument.
    // TODO: would it really still happen?
    #[arg(long)]
    nocapture: bool,
    /// Prevents the terminal from being cleared of test names and includes
    /// non-essential test messages.
    #[arg(short, long)]
    verbose: bool,
}

/// Which things to print out for debugging.


impl Args {
    fn matches(&self, canonicalized_path: &Path) -> bool {
        let path = canonicalized_path.to_string_lossy();
        if !self.exact {
            return self.filter.is_empty()
                || self.filter.iter().any(|v| path.contains(v));
        }

        self.filter.iter().any(|v| match path.strip_suffix(v) {
            None => false,
            Some(residual) => {
                residual.is_empty() || residual.ends_with(MAIN_SEPARATOR_STR)
            }
        })
    }
}

/// Tests all test files and prints a summary.
fn main() {
    let args = Args::parse();

    // Create loader and context.
    let world = TestWorld::new(args.print);

    println!("Running tests...");
    let results = WalkDir::new(TYP_DIR)
        .into_iter()
        .par_bridge()
        .filter_map(|entry| {
            let entry = entry.unwrap();
            if entry.depth() == 0 {
                return None;
            }

            if entry.path().starts_with("typ/benches") {
                return None;
            }

            let src_path = entry.into_path(); // Relative to TYP_DIR.
            if src_path.extension() != Some(OsStr::new("typ")) {
                return None;
            }

            if args.matches(&src_path.canonicalize().unwrap()) {
                Some(src_path)
            } else {
                None
            }
        })
        .map_with(world, |world, src_path| {
            let path = src_path.strip_prefix(TYP_DIR).unwrap();
            let png_path = Path::new(PNG_DIR).join(path).with_extension("png");
            let ref_path = Path::new(REF_DIR).join(path).with_extension("png");
            let svg_path = Path::new(SVG_DIR).join(path).with_extension("svg");
            let pdf_path =
                args.pdf.then(|| Path::new(PDF_DIR).join(path).with_extension("pdf"));

            test(
                world,
                &src_path,
                &png_path,
                &ref_path,
                pdf_path.as_deref(),
                &svg_path,
                &args,
            ) as usize
        })
        .collect::<Vec<_>>();

    let len = results.len();
    let ok = results.iter().sum::<usize>();
    if len > 0 {
        println!("{ok} / {len} test{} passed.", if len > 1 { "s" } else { "" });
    } else {
        println!("No test ran.");
    }

    if ok != len {
        println!(
            "Set the UPDATE_EXPECT environment variable or pass the \
             --update flag to update the reference image(s)."
        );
    }

    if ok < len {
        std::process::exit(1);
    }
}



/// A world that provides access to the tests environment.


/// The file system path for a file ID.


/// Read a file.


/// Tests a test file and prints the result.
///
/// Also tests that the header of each test is written correctly.
/// See [parse_part_metadata] for more details.
fn test(
    world: &mut TestWorld,
    src_path: &Path,
    png_path: &Path,
    ref_path: &Path,
    pdf_path: Option<&Path>,
    svg_path: &Path,
    args: &Args,
) -> bool {
    struct PanicGuard<'a>(&'a Path);
    impl Drop for PanicGuard<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                println!("Panicked in {}", self.0.display());
            }
        }
    }

    let name = src_path.strip_prefix(TYP_DIR).unwrap_or(src_path);
    let text = fs::read_to_string(src_path).unwrap();
    let _guard = PanicGuard(name);

    let mut output = String::new();
    let mut ok = true;
    let mut updated = false;
    let mut pages = vec![];
    let mut line = 0;
    let mut header_configuration = None;
    let mut compare_ever = false;
    let mut rng = LinearShift::new();

    let parts: Vec<_> = text
        .split("\n---")
        .map(|s| s.strip_suffix('\r').unwrap_or(s))
        .collect();

    for (i, &part) in parts.iter().enumerate() {
        if let Some(x) = args.subtest {
            let x = usize::try_from(
                x.rem_euclid(isize::try_from(parts.len()).unwrap_or_default()),
            )
            .unwrap();
            if x != i {
                writeln!(output, "  Skipped subtest {i}.").unwrap();
                continue;
            }
        }
        let is_header = i == 0
            && parts.len() > 1
            && part
                .lines()
                .all(|s| s.starts_with("//") || s.chars().all(|c| c.is_whitespace()));

        if is_header {
            let source = Source::detached(part.to_string());
            let metadata = parse_part_metadata(&source, true);
            match metadata {
                Ok(metadata) => {
                    header_configuration = Some(metadata.config);
                }
                Err(invalid_data) => {
                    ok = false;
                    writeln!(
                        output,
                        " Test {}: invalid metadata in header, failing the test:",
                        name.display()
                    )
                    .unwrap();
                    InvalidMetadata::write(
                        invalid_data,
                        &mut output,
                        &mut |annotation, output| {
                            print_annotation(output, &source, line, annotation)
                        },
                    );
                }
            }
        } else {
            let (part_ok, compare_here, part_frames) = test_part(
                &mut output,
                world,
                src_path,
                part.into(),
                line,
                i,
                header_configuration.as_ref().unwrap_or(&Default::default()),
                &mut rng,
                args.verbose,
            );

            ok &= part_ok;
            compare_ever |= compare_here;
            pages.extend(part_frames);
        }

        line += part.lines().count() + 1;
    }

    let document = Document { pages, ..Default::default() };
    if compare_ever {
        if let Some(pdf_path) = pdf_path {
            let pdf_data = typst_pdf::pdf(
                &document,
                Some(&format!("typst-test: {}", name.display())),
                world.today(Some(0)),
            );
            fs::create_dir_all(pdf_path.parent().unwrap()).unwrap();
            fs::write(pdf_path, pdf_data).unwrap();
        }

        if world.print.frames {
            for frame in &document.pages {
                writeln!(output, "{frame:#?}\n").unwrap();
            }
        }

        let canvas = render(&document);
        fs::create_dir_all(png_path.parent().unwrap()).unwrap();
        canvas.save_png(png_path).unwrap();

        let svg = typst_svg::svg_merged(&document, Abs::pt(5.0));

        fs::create_dir_all(svg_path.parent().unwrap()).unwrap();
        std::fs::write(svg_path, svg.as_bytes()).unwrap();

        if let Ok(ref_pixmap) = sk::Pixmap::load_png(ref_path) {
            if canvas.width() != ref_pixmap.width()
                || canvas.height() != ref_pixmap.height()
                || canvas
                    .data()
                    .iter()
                    .zip(ref_pixmap.data())
                    .any(|(&a, &b)| a.abs_diff(b) > 2)
            {
                if args.update {
                    update_image(png_path, ref_path);
                    updated = true;
                } else {
                    writeln!(output, "  Does not match reference image.").unwrap();
                    ok = false;
                }
            }
        } else if !document.pages.is_empty() {
            if args.update {
                update_image(png_path, ref_path);
                updated = true;
            } else {
                writeln!(output, "  Failed to open reference image.").unwrap();
                ok = false;
            }
        }
    }

    {
        let mut stdout = io::stdout().lock();
        stdout.write_all(name.to_string_lossy().as_bytes()).unwrap();
        if ok {
            writeln!(stdout, " ✔").unwrap();
            // Don't clear the line when in verbose mode or when the reference image
            // was updated, to show in the output which test had its image updated.
            if !updated && !args.verbose && stdout.is_terminal() {
                // ANSI escape codes: cursor moves up and clears the line.
                write!(stdout, "\x1b[1A\x1b[2K").unwrap();
            }
        } else {
            writeln!(stdout, " ❌").unwrap();
        }
        if updated {
            writeln!(stdout, "  Updated reference image.").unwrap();
        }
        if !output.is_empty() {
            stdout.write_all(output.as_bytes()).unwrap();
        }
    }

    ok
}

fn update_image(png_path: &Path, ref_path: &Path) {
    oxipng::optimize(
        &InFile::Path(png_path.to_owned()),
        &OutFile::from_path(ref_path.to_owned()),
        &Options::max_compression(),
    )
    .unwrap();
}

#[allow(clippy::too_many_arguments)]
fn test_part(
    output: &mut String,
    world: &mut TestWorld,
    src_path: &Path,
    text: String,
    line: usize,
    i: usize,
    header_configuration: &TestConfig,
    rng: &mut LinearShift,
    verbose: bool,
) -> (bool, bool, Vec<Page>) {
    let source = world.set(src_path, text);
    if world.print.syntax {
        writeln!(output, "Syntax Tree:\n{:#?}\n", source.root()).unwrap();
    }

    if world.print.model {
        print_model(world, &source, output);
    }

    let mut tracer = Tracer::new();
    let (mut frames, diagnostics) = match typst::compile(world, &mut tracer) {
        Ok(document) => (document.pages, tracer.warnings()),
        Err(errors) => {
            let mut warnings = tracer.warnings();
            warnings.extend(errors);
            (vec![], warnings)
        }
    };

    let metadata = parse_part_metadata(&source, false);
    match metadata {
        Ok(metadata) => {
            let mut ok = true;
            let compare_ref = metadata
                .config
                .compare_ref
                .unwrap_or(header_configuration.compare_ref.unwrap_or(true));
            let validate_hints = metadata
                .config
                .validate_hints
                .unwrap_or(header_configuration.validate_hints.unwrap_or(true));
            let validate_autocomplete = metadata
                .config
                .validate_autocomplete
                .unwrap_or(header_configuration.validate_autocomplete.unwrap_or(false));

            if verbose {
                writeln!(output, "Subtest {i} runs with compare_ref={compare_ref}; validate_hints={validate_hints}; validate_autocomplete={validate_autocomplete};").unwrap();
            }
            ok &= test_spans(output, source.root());
            ok &= test_reparse(output, source.text(), i, rng);

            // Don't retain frames if we don't want to compare with reference images.
            if !compare_ref {
                frames.clear();
            }

            // we never check autocomplete and error at the same time

            let diagnostic_annotations = metadata
                .annotations
                .iter()
                .filter(|a| {
                    !matches!(
                        a.kind,
                        AnnotationKind::AutocompleteContains
                            | AnnotationKind::AutocompleteExcludes
                    )
                })
                .cloned()
                .collect::<HashSet<_>>();

            if validate_autocomplete {
                // warns and ignores diagnostics
                if !diagnostic_annotations.is_empty() {
                    writeln!(
                        output,
                        "  Subtest {i} contains diagnostics but is in autocomplete mode."
                    )
                    .unwrap();
                    for annotation in diagnostic_annotations {
                        write!(output, "    Ignored | ").unwrap();
                        print_annotation(output, &source, line, &annotation);
                    }
                }

                test_autocomplete(
                    output,
                    world,
                    &source,
                    line,
                    i,
                    &mut ok,
                    metadata.annotations.iter(),
                );
            } else {
                test_diagnostics(
                    output,
                    world,
                    &source,
                    line,
                    i,
                    &mut ok,
                    validate_hints,
                    diagnostics.iter(),
                    &diagnostic_annotations,
                );
            }

            (ok, compare_ref, frames)
        }
        Err(invalid_data) => {
            writeln!(output, "  Subtest {i} has invalid metadata, failing the test:")
                .unwrap();
            InvalidMetadata::write(
                invalid_data,
                output,
                &mut |annotation: &Annotation, output: &mut String| {
                    print_annotation(output, &source, line, annotation)
                },
            );

            (false, false, frames)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn test_autocomplete<'a>(
    output: &mut String,
    world: &mut TestWorld,
    source: &Source,
    line: usize,
    i: usize,
    ok: &mut bool,
    annotations: impl Iterator<Item = &'a Annotation>,
) {
    for annotation in annotations.filter(|a| {
        matches!(
            a.kind,
            AnnotationKind::AutocompleteContains | AnnotationKind::AutocompleteExcludes
        )
    }) {
        // Ok cause we checked in parsing that range was Some for this annotation
        let cursor = annotation.range.as_ref().unwrap().start;

        // todo, use document if is_some to test labels autocomplete
        let completions = typst_ide::autocomplete(world, None, source, cursor, true)
            .map(|(_, c)| c)
            .unwrap_or_default()
            .into_iter()
            .map(|c| c.label.to_string())
            .collect::<HashSet<_>>();
        let completions =
            completions.iter().map(|s| s.as_str()).collect::<HashSet<&str>>();

        let must_contain_or_exclude = parse_string_list(&annotation.text);
        let missing =
            must_contain_or_exclude.difference(&completions).collect::<Vec<_>>();

        if !missing.is_empty()
            && matches!(annotation.kind, AnnotationKind::AutocompleteContains)
        {
            writeln!(output, "  Subtest {i} does not match expected completions.")
                .unwrap();
            write!(output, "  for annotation | ").unwrap();
            print_annotation(output, source, line, annotation);

            write!(output, "    Not contained  | ").unwrap();
            for item in missing {
                write!(output, "{item:?}, ").unwrap()
            }
            writeln!(output).unwrap();
            *ok = false;
        }

        let undesired =
            must_contain_or_exclude.intersection(&completions).collect::<Vec<_>>();

        if !undesired.is_empty()
            && matches!(annotation.kind, AnnotationKind::AutocompleteExcludes)
        {
            writeln!(output, "  Subtest {i} does not match expected completions.")
                .unwrap();
            write!(output, "  for annotation | ").unwrap();
            print_annotation(output, source, line, annotation);

            write!(output, "    Not excluded| ").unwrap();
            for item in undesired {
                write!(output, "{item:?}, ").unwrap()
            }
            writeln!(output).unwrap();
            *ok = false;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn test_diagnostics<'a>(
    output: &mut String,
    world: &mut TestWorld,
    source: &Source,
    line: usize,
    i: usize,
    ok: &mut bool,
    validate_hints: bool,
    diagnostics: impl Iterator<Item = &'a SourceDiagnostic>,
    diagnostic_annotations: &HashSet<Annotation>,
) {
    // Map diagnostics to range and message format, discard traces and errors from
    // other files, collect hints.
    //
    // This has one caveat: due to the format of the expected hints, we can not
    // verify if a hint belongs to a diagnostic or not. That should be irrelevant
    // however, as the line of the hint is still verified.
    let mut actual_diagnostics = HashSet::new();
    for diagnostic in diagnostics {
        // Ignore diagnostics from other files.
        if diagnostic.span.id().map_or(false, |id| id != source.id()) {
            continue;
        }

        let annotation = Annotation {
            kind: match diagnostic.severity {
                Severity::Error => AnnotationKind::Error,
                Severity::Warning => AnnotationKind::Warning,
            },
            range: world.range(diagnostic.span),
            text: diagnostic.message.replace("\\", "/"),
        };

        if validate_hints {
            for hint in &diagnostic.hints {
                actual_diagnostics.insert(Annotation {
                    kind: AnnotationKind::Hint,
                    text: hint.clone(),
                    range: annotation.range.clone(),
                });
            }
        }

        actual_diagnostics.insert(annotation);
    }

    // Basically symmetric_difference, but we need to know where an item is coming from.
    let mut unexpected_outputs = actual_diagnostics
        .difference(diagnostic_annotations)
        .collect::<Vec<_>>();
    let mut missing_outputs = diagnostic_annotations
        .difference(&actual_diagnostics)
        .collect::<Vec<_>>();

    unexpected_outputs.sort_by_key(|&v| v.range.as_ref().map(|r| r.start));
    missing_outputs.sort_by_key(|&v| v.range.as_ref().map(|r| r.start));

    // This prints all unexpected emits first, then all missing emits.
    // Is this reasonable or subject to change?
    if !(unexpected_outputs.is_empty() && missing_outputs.is_empty()) {
        writeln!(output, "  Subtest {i} does not match expected errors.").unwrap();
        *ok = false;

        for unexpected in unexpected_outputs {
            write!(output, "    Not annotated | ").unwrap();
            print_annotation(output, source, line, unexpected)
        }

        for missing in missing_outputs {
            write!(output, "    Not emitted   | ").unwrap();
            print_annotation(output, source, line, missing)
        }
    }
}

fn print_model(world: &mut TestWorld, source: &Source, output: &mut String) {
    let world = (world as &dyn World).track();
    let route = typst::engine::Route::default();
    let mut tracer = typst::eval::Tracer::new();

    let module =
        typst::eval::eval(world, route.track(), tracer.track_mut(), source).unwrap();
    writeln!(output, "Model:\n{:#?}\n", module.content()).unwrap();
}

fn print_annotation(
    output: &mut String,
    source: &Source,
    line: usize,
    annotation: &Annotation,
) {
    let Annotation { range, text, kind } = annotation;
    write!(output, "{kind}: ").unwrap();
    if let Some(range) = range {
        let start_line = 1 + line + source.byte_to_line(range.start).unwrap();
        let start_col = 1 + source.byte_to_column(range.start).unwrap();
        let end_line = 1 + line + source.byte_to_line(range.end).unwrap();
        let end_col = 1 + source.byte_to_column(range.end).unwrap();
        write!(output, "{start_line}:{start_col}-{end_line}:{end_col}: ").unwrap();
    }
    writeln!(output, "{text}").unwrap();
}

/// Pseudorandomly edit the source file and test whether a reparse produces the
/// same result as a clean parse.
///
/// The method will first inject 10 strings once every 400 source characters
/// and then select 5 leaf node boundaries to inject an additional, randomly
/// chosen string from the injection list.
fn test_reparse(
    output: &mut String,
    text: &str,
    i: usize,
    rng: &mut LinearShift,
) -> bool {
    let supplements = [
        "[",
        "]",
        "{",
        "}",
        "(",
        ")",
        "#rect()",
        "a word",
        ", a: 1",
        "10.0",
        ":",
        "if i == 0 {true}",
        "for",
        "* hello *",
        "//",
        "/*",
        "\\u{12e4}",
        "```typst",
        " ",
        "trees",
        "\\",
        "$ a $",
        "2.",
        "-",
        "5",
    ];

    let mut ok = true;
    let mut apply = |replace: Range<usize>, with| {
        let mut incr_source = Source::detached(text);
        if incr_source.root().len() != text.len() {
            println!(
                "    Subtest {i} tree length {} does not match string length {} ❌",
                incr_source.root().len(),
                text.len(),
            );
            return false;
        }

        incr_source.edit(replace.clone(), with);

        let edited_src = incr_source.text();
        let ref_source = Source::detached(edited_src);
        let ref_root = ref_source.root();
        let incr_root = incr_source.root();

        // Ensures that the span numbering invariants hold.
        let spans_ok = test_spans(output, ref_root) && test_spans(output, incr_root);

        // Ensure that the reference and incremental trees are the same.
        let tree_ok = ref_root.spanless_eq(incr_root);

        if !tree_ok {
            writeln!(
                output,
                "    Subtest {i} reparse differs from clean parse when inserting '{with}' at {}-{} ❌\n",
                replace.start, replace.end,
            ).unwrap();
            writeln!(output, "    Expected reference tree:\n{ref_root:#?}\n").unwrap();
            writeln!(output, "    Found incremental tree:\n{incr_root:#?}").unwrap();
            writeln!(
                output,
                "    Full source ({}):\n\"{edited_src:?}\"",
                edited_src.len()
            )
            .unwrap();
        }

        spans_ok && tree_ok
    };

    let mut pick = |range: Range<usize>| {
        let ratio = rng.next();
        (range.start as f64 + ratio * (range.end - range.start) as f64).floor() as usize
    };

    let insertions = (text.len() as f64 / 400.0).ceil() as usize;
    for _ in 0..insertions {
        let supplement = supplements[pick(0..supplements.len())];
        let start = pick(0..text.len());
        let end = pick(start..text.len());

        if !text.is_char_boundary(start) || !text.is_char_boundary(end) {
            continue;
        }

        ok &= apply(start..end, supplement);
    }

    let source = Source::detached(text);
    let leafs = leafs(source.root());
    let start = source.find(leafs[pick(0..leafs.len())].span()).unwrap().offset();
    let supplement = supplements[pick(0..supplements.len())];
    ok &= apply(start..start, supplement);

    ok
}

/// Returns all leaf descendants of a node (may include itself).
fn leafs(node: &SyntaxNode) -> Vec<SyntaxNode> {
    if node.children().len() == 0 {
        vec![node.clone()]
    } else {
        node.children().flat_map(leafs).collect()
    }
}

/// Ensure that all spans are properly ordered (and therefore unique).
#[track_caller]
fn test_spans(output: &mut String, root: &SyntaxNode) -> bool {
    test_spans_impl(output, root, 0..u64::MAX)
}

#[track_caller]
fn test_spans_impl(output: &mut String, node: &SyntaxNode, within: Range<u64>) -> bool {
    if !within.contains(&node.span().number()) {
        writeln!(output, "    Node: {node:#?}").unwrap();
        writeln!(
            output,
            "    Wrong span order: {} not in {within:?} ❌",
            node.span().number()
        )
        .unwrap();
    }

    let start = node.span().number() + 1;
    let mut children = node.children().peekable();
    while let Some(child) = children.next() {
        let end = children.peek().map_or(within.end, |next| next.span().number());
        if !test_spans_impl(output, child, start..end) {
            return false;
        }
    }

    true
}







/// A Linear-feedback shift register using XOR as its shifting function.
/// Can be used as PRNG.
struct LinearShift(u64);

impl LinearShift {
    /// Initialize the shift register with a pre-set seed.
    pub fn new() -> Self {
        Self(0xACE5)
    }

    /// Return a pseudo-random number between `0.0` and `1.0`.
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 3;
        self.0 ^= self.0 << 14;
        self.0 ^= self.0 >> 28;
        self.0 ^= self.0 << 36;
        self.0 ^= self.0 >> 52;
        self.0 as f64 / u64::MAX as f64
    }
}

/// Parse metadata for a test.
pub fn parse_part_metadata(
    source: &Source,
    is_header: bool,
) -> Result<TestMetadata, Vec<InvalidMetadata>> {
    let mut config = TestConfig::default();
    let mut annotations = HashSet::default();
    let mut invalid_data = vec![];

    let lines = source_to_lines(source);

    for (i, line) in lines.iter().enumerate() {
        if let Some((key, value)) = parse_metadata_line(line) {
            let key = key.trim();
            match key {
                "Ref" => validate_set_annotation(
                    value,
                    &mut config.compare_ref,
                    &mut invalid_data,
                ),
                "Hints" => validate_set_annotation(
                    value,
                    &mut config.validate_hints,
                    &mut invalid_data,
                ),
                "Autocomplete" => validate_set_annotation(
                    value,
                    &mut config.validate_autocomplete,
                    &mut invalid_data,
                ),
                annotation_key => {
                    let Ok(kind) = AnnotationKind::from_str(annotation_key) else {
                        continue;
                    };
                    let mut s = Scanner::new(value);
                    let range = parse_range(&mut s, i, source);
                    let rest = if range.is_some() { s.after() } else { s.string() };
                    let message = rest
                        .trim()
                        .replace("VERSION", &PackageVersion::compiler().to_string())
                        .into();

                    let annotation =
                        Annotation { kind, range: range.clone(), text: message };

                    if is_header {
                        invalid_data.push(InvalidMetadata::InvalidAnnotation(
                            annotation,
                            format!(
                                "Error: header may not contain annotations of type {kind}"
                            ),
                        ));
                        continue;
                    }

                    if matches!(
                        kind,
                        AnnotationKind::AutocompleteContains
                            | AnnotationKind::AutocompleteExcludes
                    ) {
                        if let Some(range) = range {
                            if range.start != range.end {
                                invalid_data.push(InvalidMetadata::InvalidAnnotation(
                                    annotation,
                                    "Error: found range in Autocomplete annotation where range.start != range.end, range.end would be ignored."
                                        .to_string()
                                    ));
                                continue;
                            }
                        } else {
                            invalid_data.push(InvalidMetadata::InvalidAnnotation(
                                annotation,
                                "Error: autocomplete annotation but no range specified"
                                    .to_string(),
                            ));
                            continue;
                        }
                    }
                    annotations.insert(annotation);
                }
            }
        }
    }
    if invalid_data.is_empty() {
        Ok(TestMetadata { config, annotations })
    } else {
        Err(invalid_data)
    }
}

/// Extract key and value for a metadata line of the form: `// KEY: VALUE`.
fn parse_metadata_line(line: &str) -> Option<(&str, &str)> {
    let mut s = Scanner::new(line);
    if !s.eat_if("// ") {
        return None;
    }

    let key = s.eat_until(':').trim();
    if !s.eat_if(':') {
        return None;
    }

    let value = s.eat_until('\n').trim();
    Some((key, value))
}

/// Parse a quoted string.
fn parse_string<'a>(s: &mut Scanner<'a>) -> Option<&'a str> {
    if !s.eat_if('"') {
        return None;
    }
    let sub = s.eat_until('"');
    if !s.eat_if('"') {
        return None;
    }

    Some(sub)
}

/// Parse a number.
fn parse_num(s: &mut Scanner) -> Option<isize> {
    let mut first = true;
    let n = &s.eat_while(|c: char| {
        let valid = first && c == '-' || c.is_numeric();
        first = false;
        valid
    });
    n.parse().ok()
}

/// Parse a comma-separated list of strings.
pub fn parse_string_list(text: &str) -> HashSet<&str> {
    let mut s = Scanner::new(text);
    let mut result = HashSet::new();
    while let Some(sub) = parse_string(&mut s) {
        result.insert(sub);
        s.eat_whitespace();
        if !s.eat_if(',') {
            break;
        }
        s.eat_whitespace();
    }
    result
}

/// Parse a position.
fn parse_pos(s: &mut Scanner, i: usize, source: &Source) -> Option<usize> {
    let first = parse_num(s)? - 1;
    let (delta, column) =
        if s.eat_if(':') { (first, parse_num(s)? - 1) } else { (0, first) };
    let line = (i + comments_until_code(source, i)).checked_add_signed(delta)?;
    source.line_column_to_byte(line, usize::try_from(column).ok()?)
}

/// Parse a range.
fn parse_range(s: &mut Scanner, i: usize, source: &Source) -> Option<Range<usize>> {
    let lines = source_to_lines(source);
    s.eat_whitespace();
    if s.eat_if("-1") {
        let mut add = 1;
        while let Some(line) = lines.get(i + add) {
            if !line.starts_with("//") {
                break;
            }
            add += 1;
        }
        let next_line = lines.get(i + add)?;
        let col = next_line.chars().count();

        let index = source.line_column_to_byte(i + add, col)?;
        s.eat_whitespace();
        return Some(index..index);
    }
    let start = parse_pos(s, i, source)?;
    let end = if s.eat_if('-') { parse_pos(s, i, source)? } else { start };
    s.eat_whitespace();
    Some(start..end)
}

/// Returns the number of lines of comment from line i to next line of code.
fn comments_until_code(source: &Source, i: usize) -> usize {
    source_to_lines(source)[i..]
        .iter()
        .take_while(|line| line.starts_with("//"))
        .count()
}

fn source_to_lines(source: &Source) -> Vec<&str> {
    source.text().lines().map(str::trim).collect()
}

fn validate_set_annotation(
    value: &str,
    flag: &mut Option<bool>,
    invalid_data: &mut Vec<InvalidMetadata>,
) {
    let value = value.trim();
    if value != "false" && value != "true" {
        invalid_data.push(
            InvalidMetadata::InvalidSet(format!("Error: trying to set Ref, Hints, or Autocomplete with value {value:?} != true, != false.")))
    } else {
        *flag = Some(value == "true")
    }
}
