#include "XFeatVocabulary.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace
{

struct Options
{
    std::string vocabularyPath;
    std::vector<std::string> inputs;
    std::string pairsPath;
    std::string outputPath;
    std::vector<std::string> groundTruthSpecs;
    int maxFrames = 0;
    int maxFeaturesPerFrame = 512;
    int queryStep = 1;
    int topK = 20;
    int excludeNearbyFrames = 30;
    double gtMaxTimeDiff = 0.02;
    double gtMinTimeGap = 30.0;
    double gtLoopDistance = 2.0;
    double gtMaxAngleDeg = 180.0;
    bool printTopRows = true;
};

struct FrameEntry
{
    int globalId = -1;
    int sequenceId = -1;
    int localId = -1;
    std::string sequence;
    std::string stamp;
    double timestamp = 0.0;
    std::string image;
    int descriptors = 0;
    bool hasPose = false;
    cv::Vec3d position = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec4d quaternion = cv::Vec4d(1.0, 0.0, 0.0, 0.0); // qw, qx, qy, qz
    DBoW3::BowVector bow;
};

struct Candidate
{
    int frameId = -1;
    double score = 0.0;
};

struct PairLabel
{
    int query = -1;
    int target = -1;
    std::string label;
};

struct PoseEntry
{
    double timestamp = 0.0;
    cv::Vec3d position = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec4d quaternion = cv::Vec4d(1.0, 0.0, 0.0, 0.0); // qw, qx, qy, qz
};

constexpr double kPi = 3.14159265358979323846;

std::string BaseName(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const std::string ymlGz = ".yml.gz";
    if(name.size() > ymlGz.size() && name.compare(name.size() - ymlGz.size(), ymlGz.size(), ymlGz) == 0)
        name.resize(name.size() - ymlGz.size());
    else
    {
        const size_t dot = name.find_last_of('.');
        if(dot != std::string::npos)
            name.resize(dot);
    }
    return name;
}

void PrintUsage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <vocabulary.dbow3|vocabulary.yml.gz> <descriptors.yml.gz> [more.yml.gz] [options]\n\n"
        << "Options:\n"
        << "  --pairs <file>                  Optional labels: '<query> <target> [label]' per line\n"
        << "                                  query/target can be global ids or sequence:local_id\n"
        << "  --output <csv>                  Optional CSV output path\n"
        << "  --gt <file|sequence:file>       Optional EuRoC CSV or TUM trajectory for GT true-loop labels\n"
        << "                                  Repeat for multiple sequences\n"
        << "  --gt-max-time-diff <sec>        Max timestamp association error. Default: 0.02\n"
        << "  --gt-min-time-gap <sec>         True-loop minimum time separation. Default: 30\n"
        << "  --gt-loop-distance <m>          True-loop maximum position distance. Default: 2.0\n"
        << "  --gt-max-angle-deg <deg>        Optional orientation gate. Default: 180\n"
        << "  --max-frames <N>                0 means no limit. Default: 0\n"
        << "  --max-features-per-frame <N>    0 means all descriptors. Default: 512\n"
        << "  --query-step <N>                Evaluate every Nth frame. Default: 1\n"
        << "  --top-k <N>                     Number of candidates to report. Default: 20\n"
        << "  --exclude-nearby-frames <N>     Skip same-sequence frames within this local-id gap. Default: 30\n"
        << "  --print-top-rows <0|1>          Print per-query top candidates to stdout. Default: 1\n";
}

int ParseInt(const std::string& value, const char* name)
{
    try
    {
        return std::stoi(value);
    }
    catch(...)
    {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + value);
    }
}

double ParseDouble(const std::string& value, const char* name)
{
    try
    {
        return std::stod(value);
    }
    catch(...)
    {
        throw std::runtime_error(std::string("invalid number for ") + name + ": " + value);
    }
}

std::string RequireValue(int& i, int argc, char** argv, const char* option)
{
    if(i + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + option);
    ++i;
    return argv[i];
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    std::vector<std::string> positional;

    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--pairs")
            options.pairsPath = RequireValue(i, argc, argv, "--pairs");
        else if(arg == "--output")
            options.outputPath = RequireValue(i, argc, argv, "--output");
        else if(arg == "--gt" || arg == "--groundtruth")
            options.groundTruthSpecs.push_back(RequireValue(i, argc, argv, arg.c_str()));
        else if(arg == "--gt-max-time-diff")
            options.gtMaxTimeDiff = std::max(0.0, ParseDouble(RequireValue(i, argc, argv, "--gt-max-time-diff"), "--gt-max-time-diff"));
        else if(arg == "--gt-min-time-gap")
            options.gtMinTimeGap = std::max(0.0, ParseDouble(RequireValue(i, argc, argv, "--gt-min-time-gap"), "--gt-min-time-gap"));
        else if(arg == "--gt-loop-distance")
            options.gtLoopDistance = std::max(0.0, ParseDouble(RequireValue(i, argc, argv, "--gt-loop-distance"), "--gt-loop-distance"));
        else if(arg == "--gt-max-angle-deg")
            options.gtMaxAngleDeg = std::max(0.0, ParseDouble(RequireValue(i, argc, argv, "--gt-max-angle-deg"), "--gt-max-angle-deg"));
        else if(arg == "--max-frames")
            options.maxFrames = std::max(0, ParseInt(RequireValue(i, argc, argv, "--max-frames"), "--max-frames"));
        else if(arg == "--max-features-per-frame")
            options.maxFeaturesPerFrame = std::max(0, ParseInt(RequireValue(i, argc, argv, "--max-features-per-frame"), "--max-features-per-frame"));
        else if(arg == "--query-step")
            options.queryStep = std::max(1, ParseInt(RequireValue(i, argc, argv, "--query-step"), "--query-step"));
        else if(arg == "--top-k")
            options.topK = std::max(1, ParseInt(RequireValue(i, argc, argv, "--top-k"), "--top-k"));
        else if(arg == "--exclude-nearby-frames")
            options.excludeNearbyFrames = std::max(0, ParseInt(RequireValue(i, argc, argv, "--exclude-nearby-frames"), "--exclude-nearby-frames"));
        else if(arg == "--print-top-rows")
            options.printTopRows = ParseInt(RequireValue(i, argc, argv, "--print-top-rows"), "--print-top-rows") != 0;
        else if(arg == "--help" || arg == "-h")
        {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        else if(!arg.empty() && arg[0] == '-')
            throw std::runtime_error("unknown option: " + arg);
        else
            positional.push_back(arg);
    }

    if(positional.size() < 2)
    {
        PrintUsage(argv[0]);
        throw std::runtime_error("expected vocabulary and at least one descriptor file");
    }

    options.vocabularyPath = positional.front();
    options.inputs.assign(positional.begin() + 1, positional.end());
    return options;
}

double NormalizeTimestamp(const double timestamp)
{
    return std::abs(timestamp) > 1e12 ? timestamp * 1e-9 : timestamp;
}

std::vector<std::string> SplitCommaLine(const std::string& line)
{
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while(std::getline(ss, token, ','))
        tokens.push_back(token);
    return tokens;
}

std::vector<PoseEntry> LoadGroundTruthPoses(const std::string& path)
{
    std::ifstream file(path.c_str());
    if(!file.is_open())
        throw std::runtime_error("failed to open GT file: " + path);

    std::vector<PoseEntry> poses;
    std::string line;
    int lineNo = 0;
    while(std::getline(file, line))
    {
        ++lineNo;
        const size_t comment = line.find('#');
        if(comment != std::string::npos)
            line.resize(comment);
        if(line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;

        try
        {
            PoseEntry pose;
            if(line.find(',') != std::string::npos)
            {
                const std::vector<std::string> tokens = SplitCommaLine(line);
                if(tokens.size() < 8)
                    continue;

                pose.timestamp = NormalizeTimestamp(std::stod(tokens[0]));
                pose.position = cv::Vec3d(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
                pose.quaternion = cv::Vec4d(std::stod(tokens[4]),
                                            std::stod(tokens[5]),
                                            std::stod(tokens[6]),
                                            std::stod(tokens[7]));
            }
            else
            {
                std::stringstream ss(line);
                std::vector<std::string> tokens;
                std::string token;
                while(ss >> token)
                    tokens.push_back(token);
                if(tokens.size() < 8)
                    continue;

                pose.timestamp = NormalizeTimestamp(std::stod(tokens[0]));
                pose.position = cv::Vec3d(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
                pose.quaternion = cv::Vec4d(std::stod(tokens[7]),
                                            std::stod(tokens[4]),
                                            std::stod(tokens[5]),
                                            std::stod(tokens[6]));
            }
            poses.push_back(pose);
        }
        catch(const std::exception&)
        {
            throw std::runtime_error("invalid GT line " + std::to_string(lineNo) + " in " + path);
        }
    }

    std::sort(poses.begin(), poses.end(), [](const PoseEntry& a, const PoseEntry& b) {
        return a.timestamp < b.timestamp;
    });
    return poses;
}

cv::Mat SampleRowsEvenly(const cv::Mat& descriptors, int maxRows)
{
    if(maxRows <= 0 || descriptors.rows <= maxRows)
        return descriptors.clone();

    cv::Mat sampled(maxRows, descriptors.cols, descriptors.type());
    if(maxRows == 1)
    {
        descriptors.row(0).copyTo(sampled.row(0));
        return sampled;
    }

    const double scale = static_cast<double>(descriptors.rows - 1) / static_cast<double>(maxRows - 1);
    for(int i = 0; i < maxRows; ++i)
    {
        const int row = std::min(descriptors.rows - 1, static_cast<int>(std::lround(i * scale)));
        descriptors.row(row).copyTo(sampled.row(i));
    }
    return sampled;
}

void LoadDescriptorFile(const std::string& path,
                        const int sequenceId,
                        const Options& options,
                        const ORB_SLAM3::XFeatVocabulary& vocabulary,
                        std::vector<FrameEntry>& frames)
{
    const std::string sequence = BaseName(path);
    const int startFrames = static_cast<int>(frames.size());
    long long descriptorsAdded = 0;

    std::cout << "[XFeatVocabEval] loading descriptors: " << path
              << " sequence=" << sequence << std::endl;

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if(!fs.isOpened())
        throw std::runtime_error("failed to open descriptor file: " + path);

    const std::string format = static_cast<std::string>(fs["format"]);
    if(format != "xfeat_descriptors_v1")
        throw std::runtime_error("unexpected descriptor file format in " + path + ": " + format);

    const cv::FileNode frameNodes = fs["frames"];
    if(frameNodes.type() != cv::FileNode::SEQ)
        throw std::runtime_error("descriptor file has no frame sequence: " + path);

    int localId = 0;
    for(cv::FileNodeIterator it = frameNodes.begin(); it != frameNodes.end(); ++it, ++localId)
    {
        if(options.maxFrames > 0 && static_cast<int>(frames.size()) >= options.maxFrames)
            return;

        cv::Mat descriptors;
        (*it)["descriptors"] >> descriptors;
        if(descriptors.empty())
            continue;
        if(descriptors.type() != CV_32F)
            throw std::runtime_error("descriptor matrix must be CV_32F in " + path);
        if(descriptors.cols != 64)
            throw std::runtime_error("descriptor matrix must have 64 columns in " + path);

        cv::Mat sampled = SampleRowsEvenly(descriptors, options.maxFeaturesPerFrame);

        FrameEntry entry;
        entry.globalId = static_cast<int>(frames.size());
        entry.sequenceId = sequenceId;
        entry.localId = localId;
        entry.sequence = sequence;
        entry.stamp = static_cast<std::string>((*it)["stamp"]);
        entry.timestamp = static_cast<double>((*it)["timestamp"]);
        entry.image = static_cast<std::string>((*it)["image"]);
        entry.descriptors = sampled.rows;
        vocabulary.transform(sampled, entry.bow);

        if(!entry.bow.empty())
        {
            frames.push_back(entry);
            descriptorsAdded += sampled.rows;
        }
    }

    std::cout << "[XFeatVocabEval] loaded sequence=" << sequence
              << " frames_added=" << (static_cast<int>(frames.size()) - startFrames)
              << " descriptors_added=" << descriptorsAdded
              << " total_frames=" << frames.size()
              << std::endl;
}

std::vector<std::string> UniqueSequences(const std::vector<FrameEntry>& frames)
{
    std::vector<std::string> sequences;
    for(const FrameEntry& frame : frames)
    {
        if(std::find(sequences.begin(), sequences.end(), frame.sequence) == sequences.end())
            sequences.push_back(frame.sequence);
    }
    return sequences;
}

std::map<std::string, std::string> ResolveGroundTruthSpecs(const std::vector<std::string>& specs,
                                                           const std::vector<FrameEntry>& frames)
{
    std::map<std::string, std::string> result;
    if(specs.empty())
        return result;

    const std::vector<std::string> sequences = UniqueSequences(frames);
    for(const std::string& spec : specs)
    {
        const size_t colon = spec.find(':');
        bool assigned = false;
        if(colon != std::string::npos)
        {
            const std::string maybeSequence = spec.substr(0, colon);
            if(std::find(sequences.begin(), sequences.end(), maybeSequence) != sequences.end())
            {
                result[maybeSequence] = spec.substr(colon + 1);
                assigned = true;
            }
        }

        if(!assigned)
        {
            if(sequences.size() != 1)
            {
                throw std::runtime_error(
                    "GT path without sequence prefix is only allowed for single-sequence evaluation: " + spec);
            }
            result[sequences.front()] = spec;
        }
    }
    return result;
}

void AttachGroundTruth(const Options& options, std::vector<FrameEntry>& frames)
{
    const std::map<std::string, std::string> gtBySequence =
        ResolveGroundTruthSpecs(options.groundTruthSpecs, frames);
    if(gtBySequence.empty())
        return;

    int totalAttached = 0;
    int totalFramesInGtSequences = 0;

    for(const auto& sequenceAndPath : gtBySequence)
    {
        const std::string& sequence = sequenceAndPath.first;
        const std::string& path = sequenceAndPath.second;
        const std::vector<PoseEntry> poses = LoadGroundTruthPoses(path);
        if(poses.empty())
            throw std::runtime_error("GT file contains no valid poses: " + path);

        int sequenceFrames = 0;
        int attached = 0;
        for(FrameEntry& frame : frames)
        {
            if(frame.sequence != sequence)
                continue;
            ++sequenceFrames;

            const auto lower = std::lower_bound(
                poses.begin(),
                poses.end(),
                frame.timestamp,
                [](const PoseEntry& pose, const double timestamp) {
                    return pose.timestamp < timestamp;
                });

            int bestIndex = -1;
            double bestAbsDt = std::numeric_limits<double>::max();
            if(lower != poses.end())
            {
                bestIndex = static_cast<int>(std::distance(poses.begin(), lower));
                bestAbsDt = std::abs(lower->timestamp - frame.timestamp);
            }
            if(lower != poses.begin())
            {
                const auto prev = lower - 1;
                const double absDt = std::abs(prev->timestamp - frame.timestamp);
                if(absDt < bestAbsDt)
                {
                    bestIndex = static_cast<int>(std::distance(poses.begin(), prev));
                    bestAbsDt = absDt;
                }
            }

            if(bestIndex >= 0 && bestAbsDt <= options.gtMaxTimeDiff)
            {
                frame.hasPose = true;
                frame.position = poses[bestIndex].position;
                frame.quaternion = poses[bestIndex].quaternion;
                ++attached;
            }
        }

        totalAttached += attached;
        totalFramesInGtSequences += sequenceFrames;

        std::cout << "[XFeatVocabEval][GTLoad]"
                  << " sequence=" << sequence
                  << " path=" << path
                  << " poses=" << poses.size()
                  << " frames=" << sequenceFrames
                  << " attached=" << attached
                  << " max_time_diff=" << options.gtMaxTimeDiff
                  << std::endl;
    }

    std::cout << "[XFeatVocabEval][GTLoadSummary]"
              << " sequences=" << gtBySequence.size()
              << " frames=" << totalFramesInGtSequences
              << " attached=" << totalAttached
              << std::endl;
}

std::vector<Candidate> RankCandidates(const std::vector<FrameEntry>& frames,
                                      const ORB_SLAM3::XFeatVocabulary& vocabulary,
                                      const int queryId,
                                      const int excludeNearbyFrames)
{
    const FrameEntry& query = frames[queryId];
    std::vector<Candidate> candidates;
    candidates.reserve(frames.size() > 0 ? frames.size() - 1 : 0);

    for(const FrameEntry& target : frames)
    {
        if(target.globalId == query.globalId)
            continue;
        if(target.sequenceId == query.sequenceId &&
           std::abs(target.localId - query.localId) <= excludeNearbyFrames)
        {
            continue;
        }

        const double score = vocabulary.score(query.bow, target.bow);
        if(score > 0.0)
            candidates.push_back({target.globalId, score});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if(a.score != b.score)
            return a.score > b.score;
        return a.frameId < b.frameId;
    });
    return candidates;
}

double PositionDistance(const FrameEntry& a, const FrameEntry& b)
{
    const cv::Vec3d d = a.position - b.position;
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

double QuaternionAngleDeg(const FrameEntry& a, const FrameEntry& b)
{
    const double normA = std::sqrt(a.quaternion.dot(a.quaternion));
    const double normB = std::sqrt(b.quaternion.dot(b.quaternion));
    if(normA <= 0.0 || normB <= 0.0)
        return 180.0;

    double dot = a.quaternion.dot(b.quaternion) / (normA * normB);
    dot = std::min(1.0, std::max(-1.0, dot));
    dot = std::abs(dot);
    return 2.0 * std::acos(dot) * 180.0 / kPi;
}

bool IsTrueLoop(const FrameEntry& query, const FrameEntry& target, const Options& options)
{
    if(!query.hasPose || !target.hasPose)
        return false;
    if(query.sequenceId != target.sequenceId)
        return false;
    if(std::abs(query.timestamp - target.timestamp) < options.gtMinTimeGap)
        return false;
    if(PositionDistance(query, target) > options.gtLoopDistance)
        return false;
    if(options.gtMaxAngleDeg < 179.999 && QuaternionAngleDeg(query, target) > options.gtMaxAngleDeg)
        return false;
    return true;
}

std::vector<std::string> SplitLine(const std::string& line)
{
    std::stringstream ss(line);
    std::vector<std::string> tokens;
    std::string token;
    while(ss >> token)
        tokens.push_back(token);
    return tokens;
}

int ResolveFrameRef(const std::string& token,
                    const std::vector<FrameEntry>& frames,
                    const std::map<std::string, int>& sequenceLocalToGlobal)
{
    const size_t colon = token.find(':');
    if(colon != std::string::npos)
    {
        const auto it = sequenceLocalToGlobal.find(token);
        if(it == sequenceLocalToGlobal.end())
            throw std::runtime_error("unknown frame ref: " + token);
        return it->second;
    }

    const int globalId = ParseInt(token, "pair frame ref");
    if(globalId < 0 || globalId >= static_cast<int>(frames.size()))
        throw std::runtime_error("global frame ref out of range: " + token);
    return globalId;
}

std::vector<PairLabel> LoadPairs(const std::string& path, const std::vector<FrameEntry>& frames)
{
    std::map<std::string, int> sequenceLocalToGlobal;
    for(const FrameEntry& frame : frames)
        sequenceLocalToGlobal[frame.sequence + ":" + std::to_string(frame.localId)] = frame.globalId;

    std::ifstream file(path.c_str());
    if(!file.is_open())
        throw std::runtime_error("failed to open pairs file: " + path);

    std::vector<PairLabel> pairs;
    std::string line;
    int lineNo = 0;
    while(std::getline(file, line))
    {
        ++lineNo;
        const size_t comment = line.find('#');
        if(comment != std::string::npos)
            line.resize(comment);

        const std::vector<std::string> tokens = SplitLine(line);
        if(tokens.empty())
            continue;
        if(tokens.size() < 2)
            throw std::runtime_error("invalid pair line " + std::to_string(lineNo) + " in " + path);

        PairLabel pair;
        pair.query = ResolveFrameRef(tokens[0], frames, sequenceLocalToGlobal);
        pair.target = ResolveFrameRef(tokens[1], frames, sequenceLocalToGlobal);
        pair.label = tokens.size() >= 3 ? tokens[2] : "";
        pairs.push_back(pair);
    }
    return pairs;
}

int FindRank(const std::vector<Candidate>& candidates, const int targetId, double& score)
{
    for(size_t i = 0; i < candidates.size(); ++i)
    {
        if(candidates[i].frameId == targetId)
        {
            score = candidates[i].score;
            return static_cast<int>(i);
        }
    }
    score = 0.0;
    return -1;
}

std::string CandidateRef(const FrameEntry& frame)
{
    return frame.sequence + ":" + std::to_string(frame.localId);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = ParseOptions(argc, argv);

        ORB_SLAM3::XFeatVocabulary vocabulary;
        std::cout << "[XFeatVocabEval] loading vocabulary: " << options.vocabularyPath << std::endl;
        vocabulary.load(options.vocabularyPath);
        if(vocabulary.empty())
            throw std::runtime_error("loaded vocabulary is empty: " + options.vocabularyPath);
        std::cout << "[XFeatVocabEval] vocabulary words=" << vocabulary.size()
                  << " effective_levels=" << vocabulary.getEffectiveLevels()
                  << std::endl;

        std::vector<FrameEntry> frames;
        frames.reserve(4096);
        for(size_t i = 0; i < options.inputs.size(); ++i)
        {
            LoadDescriptorFile(options.inputs[i],
                               static_cast<int>(i),
                               options,
                               vocabulary,
                               frames);
            if(options.maxFrames > 0 && static_cast<int>(frames.size()) >= options.maxFrames)
                break;
        }
        if(frames.empty())
            throw std::runtime_error("no non-empty Bow vectors were loaded");

        AttachGroundTruth(options, frames);

        double totalWords = 0.0;
        int nonEmptyBow = 0;
        for(const FrameEntry& frame : frames)
        {
            totalWords += static_cast<double>(frame.bow.size());
            if(!frame.bow.empty())
                ++nonEmptyBow;
        }

        std::cout << "[XFeatVocabEval] frames=" << frames.size()
                  << " non_empty_bow=" << nonEmptyBow
                  << " avg_bow_words=" << (frames.empty() ? 0.0 : totalWords / frames.size())
                  << " top_k=" << options.topK
                  << " exclude_nearby_frames=" << options.excludeNearbyFrames
                  << std::endl;

        std::ofstream csv;
        if(!options.outputPath.empty())
        {
            csv.open(options.outputPath.c_str());
            if(!csv.is_open())
                throw std::runtime_error("failed to open output CSV: " + options.outputPath);
            csv << "kind,query_global,query_ref,target_global,target_ref,rank,score,"
                << "top1_global,top1_ref,top1_score,top2_score,topk_score,"
                << "margin_top1_top2,margin_top1_topk,label,"
                << "true_loop_count,best_true_rank,best_true_global,best_true_ref,best_true_score,"
                << "top1_is_true,top1_distance,top1_angle_deg\n";
        }

        int evaluatedQueries = 0;
        double marginTop12Sum = 0.0;
        double marginTop1KSum = 0.0;
        for(size_t qi = 0; qi < frames.size(); qi += static_cast<size_t>(options.queryStep))
        {
            const std::vector<Candidate> ranked =
                RankCandidates(frames, vocabulary, static_cast<int>(qi), options.excludeNearbyFrames);
            if(ranked.empty())
                continue;

            ++evaluatedQueries;
            const double top1 = ranked[0].score;
            const double top2 = ranked.size() > 1 ? ranked[1].score : 0.0;
            const double topKScore = ranked.size() >= static_cast<size_t>(options.topK)
                                       ? ranked[options.topK - 1].score
                                       : ranked.back().score;
            const double marginTop12 = top1 - top2;
            const double marginTop1K = top1 - topKScore;
            marginTop12Sum += marginTop12;
            marginTop1KSum += marginTop1K;

            if(options.printTopRows)
            {
                std::cout << "[XFeatVocabEval][Top]"
                          << " query_global=" << frames[qi].globalId
                          << " query_ref=" << CandidateRef(frames[qi])
                          << " top1_global=" << ranked[0].frameId
                          << " top1_ref=" << CandidateRef(frames[ranked[0].frameId])
                          << " top1_score=" << top1
                          << " top2_score=" << top2
                          << " topk_score=" << topKScore
                          << " margin_top1_top2=" << marginTop12
                          << " margin_top1_topk=" << marginTop1K
                          << std::endl;
            }

            if(csv.is_open())
            {
                const int limit = std::min<int>(options.topK, ranked.size());
                for(int rank = 0; rank < limit; ++rank)
                {
                    const FrameEntry& target = frames[ranked[rank].frameId];
                    csv << "top,"
                        << frames[qi].globalId << ','
                        << CandidateRef(frames[qi]) << ','
                        << target.globalId << ','
                        << CandidateRef(target) << ','
                        << rank << ','
                        << ranked[rank].score << ','
                        << ranked[0].frameId << ','
                        << CandidateRef(frames[ranked[0].frameId]) << ','
                        << top1 << ','
                        << top2 << ','
                        << topKScore << ','
                        << marginTop12 << ','
                        << marginTop1K << ','
                        << ",,,,,,,,"
                        << '\n';
                }
            }
        }

        std::cout << "[XFeatVocabEval][Summary]"
                  << " evaluated_queries=" << evaluatedQueries
                  << " avg_margin_top1_top2=" << (evaluatedQueries > 0 ? marginTop12Sum / evaluatedQueries : 0.0)
                  << " avg_margin_top1_topk=" << (evaluatedQueries > 0 ? marginTop1KSum / evaluatedQueries : 0.0)
                  << std::endl;

        if(!options.groundTruthSpecs.empty())
        {
            int gtPoseMissingQueries = 0;
            int gtEvaluatedQueries = 0;
            int gtQueriesWithTrueLoop = 0;
            int gtTop1True = 0;
            int gtTop1False = 0;
            int gtRecall3 = 0;
            int gtRecall6 = 0;
            int gtRecall10 = 0;
            int gtRecall20 = 0;
            int gtDetectedTrue = 0;
            int gtMissedTrue = 0;
            double gtBestRankSumDetected = 0.0;
            double gtBestRankSumWithMiss = 0.0;
            double gtTrueLoopCountSum = 0.0;

            for(size_t qi = 0; qi < frames.size(); qi += static_cast<size_t>(options.queryStep))
            {
                const FrameEntry& query = frames[qi];
                if(!query.hasPose)
                {
                    ++gtPoseMissingQueries;
                    continue;
                }
                ++gtEvaluatedQueries;

                int trueLoopCount = 0;
                for(const FrameEntry& target : frames)
                {
                    if(target.globalId == query.globalId)
                        continue;
                    if(target.sequenceId == query.sequenceId &&
                       std::abs(target.localId - query.localId) <= options.excludeNearbyFrames)
                    {
                        continue;
                    }
                    if(IsTrueLoop(query, target, options))
                        ++trueLoopCount;
                }

                if(trueLoopCount <= 0)
                    continue;

                ++gtQueriesWithTrueLoop;
                gtTrueLoopCountSum += trueLoopCount;

                const std::vector<Candidate> ranked =
                    RankCandidates(frames, vocabulary, static_cast<int>(qi), options.excludeNearbyFrames);
                int bestTrueRank = -1;
                int bestTrueFrame = -1;
                double bestTrueScore = 0.0;
                for(size_t rank = 0; rank < ranked.size(); ++rank)
                {
                    const FrameEntry& target = frames[ranked[rank].frameId];
                    if(IsTrueLoop(query, target, options))
                    {
                        bestTrueRank = static_cast<int>(rank);
                        bestTrueFrame = target.globalId;
                        bestTrueScore = ranked[rank].score;
                        break;
                    }
                }

                const int top1Id = ranked.empty() ? -1 : ranked[0].frameId;
                const double top1Score = ranked.empty() ? 0.0 : ranked[0].score;
                const double top2Score = ranked.size() > 1 ? ranked[1].score : 0.0;
                const double topKScore = ranked.size() >= static_cast<size_t>(options.topK)
                                           ? ranked[options.topK - 1].score
                                           : (ranked.empty() ? 0.0 : ranked.back().score);
                const bool top1IsTrue = top1Id >= 0 && IsTrueLoop(query, frames[top1Id], options);
                const double top1Distance = top1Id >= 0 && query.hasPose && frames[top1Id].hasPose
                                               ? PositionDistance(query, frames[top1Id])
                                               : -1.0;
                const double top1AngleDeg = top1Id >= 0 && query.hasPose && frames[top1Id].hasPose
                                               ? QuaternionAngleDeg(query, frames[top1Id])
                                               : -1.0;

                if(bestTrueRank >= 0)
                {
                    ++gtDetectedTrue;
                    gtBestRankSumDetected += bestTrueRank;
                    gtBestRankSumWithMiss += bestTrueRank;
                    if(bestTrueRank < 3)
                        ++gtRecall3;
                    if(bestTrueRank < 6)
                        ++gtRecall6;
                    if(bestTrueRank < 10)
                        ++gtRecall10;
                    if(bestTrueRank < 20)
                        ++gtRecall20;
                }
                else
                {
                    ++gtMissedTrue;
                    gtBestRankSumWithMiss += static_cast<double>(frames.size());
                }

                if(top1IsTrue)
                    ++gtTop1True;
                else
                    ++gtTop1False;

                std::cout << "[XFeatVocabEval][GT]"
                          << " query_global=" << query.globalId
                          << " query_ref=" << CandidateRef(query)
                          << " true_loops=" << trueLoopCount
                          << " best_rank=" << bestTrueRank
                          << " best_ref=" << (bestTrueFrame >= 0 ? CandidateRef(frames[bestTrueFrame]) : "")
                          << " best_score=" << bestTrueScore
                          << " top1_ref=" << (top1Id >= 0 ? CandidateRef(frames[top1Id]) : "")
                          << " top1_score=" << top1Score
                          << " top1_true=" << (top1IsTrue ? 1 : 0)
                          << " top1_distance=" << top1Distance
                          << " top1_angle_deg=" << top1AngleDeg
                          << std::endl;

                if(csv.is_open())
                {
                    csv << "gt,"
                        << query.globalId << ','
                        << CandidateRef(query) << ','
                        << bestTrueFrame << ','
                        << (bestTrueFrame >= 0 ? CandidateRef(frames[bestTrueFrame]) : "") << ','
                        << bestTrueRank << ','
                        << bestTrueScore << ','
                        << top1Id << ','
                        << (top1Id >= 0 ? CandidateRef(frames[top1Id]) : "") << ','
                        << top1Score << ','
                        << top2Score << ','
                        << topKScore << ','
                        << (top1Score - top2Score) << ','
                        << (top1Score - topKScore) << ','
                        << "gt,"
                        << trueLoopCount << ','
                        << bestTrueRank << ','
                        << bestTrueFrame << ','
                        << (bestTrueFrame >= 0 ? CandidateRef(frames[bestTrueFrame]) : "") << ','
                        << bestTrueScore << ','
                        << (top1IsTrue ? 1 : 0) << ','
                        << top1Distance << ','
                        << top1AngleDeg << '\n';
                }
            }

            std::cout << "[XFeatVocabEval][GTSummary]"
                      << " evaluated_queries=" << gtEvaluatedQueries
                      << " pose_missing_queries=" << gtPoseMissingQueries
                      << " queries_with_true_loop=" << gtQueriesWithTrueLoop
                      << " detected_true_loop=" << gtDetectedTrue
                      << " missed_true_loop=" << gtMissedTrue
                      << " avg_true_loop_count=" << (gtQueriesWithTrueLoop > 0 ? gtTrueLoopCountSum / gtQueriesWithTrueLoop : 0.0)
                      << " top1_true_rate=" << (gtQueriesWithTrueLoop > 0 ? static_cast<double>(gtTop1True) / gtQueriesWithTrueLoop : 0.0)
                      << " top1_false=" << gtTop1False
                      << " recall@3=" << (gtQueriesWithTrueLoop > 0 ? static_cast<double>(gtRecall3) / gtQueriesWithTrueLoop : 0.0)
                      << " recall@6=" << (gtQueriesWithTrueLoop > 0 ? static_cast<double>(gtRecall6) / gtQueriesWithTrueLoop : 0.0)
                      << " recall@10=" << (gtQueriesWithTrueLoop > 0 ? static_cast<double>(gtRecall10) / gtQueriesWithTrueLoop : 0.0)
                      << " recall@20=" << (gtQueriesWithTrueLoop > 0 ? static_cast<double>(gtRecall20) / gtQueriesWithTrueLoop : 0.0)
                      << " avg_best_rank_detected=" << (gtDetectedTrue > 0 ? gtBestRankSumDetected / gtDetectedTrue : 0.0)
                      << " avg_best_rank_with_miss=" << (gtQueriesWithTrueLoop > 0 ? gtBestRankSumWithMiss / gtQueriesWithTrueLoop : 0.0)
                      << std::endl;
        }

        if(!options.pairsPath.empty())
        {
            const std::vector<PairLabel> pairs = LoadPairs(options.pairsPath, frames);
            int validPairs = 0;
            int recall3 = 0;
            int recall6 = 0;
            int recall10 = 0;
            int recall20 = 0;
            double rankSum = 0.0;
            double scoreSum = 0.0;

            for(const PairLabel& pair : pairs)
            {
                const std::vector<Candidate> ranked =
                    RankCandidates(frames, vocabulary, pair.query, options.excludeNearbyFrames);
                double pairScore = 0.0;
                const int rank = FindRank(ranked, pair.target, pairScore);
                ++validPairs;
                if(rank >= 0)
                {
                    rankSum += rank;
                    scoreSum += pairScore;
                    if(rank < 3)
                        ++recall3;
                    if(rank < 6)
                        ++recall6;
                    if(rank < 10)
                        ++recall10;
                    if(rank < 20)
                        ++recall20;
                }
                else
                {
                    rankSum += static_cast<double>(frames.size());
                }

                std::cout << "[XFeatVocabEval][Pair]"
                          << " query_global=" << pair.query
                          << " query_ref=" << CandidateRef(frames[pair.query])
                          << " target_global=" << pair.target
                          << " target_ref=" << CandidateRef(frames[pair.target])
                          << " rank=" << rank
                          << " score=" << pairScore
                          << " label=" << pair.label
                          << std::endl;

                if(csv.is_open())
                {
                    const int top1Id = ranked.empty() ? -1 : ranked[0].frameId;
                    const double top1Score = ranked.empty() ? 0.0 : ranked[0].score;
                    const double top2Score = ranked.size() > 1 ? ranked[1].score : 0.0;
                    const double topKScore = ranked.size() >= static_cast<size_t>(options.topK)
                                               ? ranked[options.topK - 1].score
                                               : (ranked.empty() ? 0.0 : ranked.back().score);
                    csv << "pair,"
                        << frames[pair.query].globalId << ','
                        << CandidateRef(frames[pair.query]) << ','
                        << frames[pair.target].globalId << ','
                        << CandidateRef(frames[pair.target]) << ','
                        << rank << ','
                        << pairScore << ','
                        << top1Id << ','
                        << (top1Id >= 0 ? CandidateRef(frames[top1Id]) : "") << ','
                        << top1Score << ','
                        << top2Score << ','
                        << topKScore << ','
                        << (top1Score - top2Score) << ','
                        << (top1Score - topKScore) << ','
                        << pair.label
                        << ",,,,,,,,"
                        << '\n';
                }
            }

            std::cout << "[XFeatVocabEval][PairsSummary]"
                      << " pairs=" << validPairs
                      << " avg_rank=" << (validPairs > 0 ? rankSum / validPairs : 0.0)
                      << " avg_score=" << (validPairs > 0 ? scoreSum / validPairs : 0.0)
                      << " recall@3=" << (validPairs > 0 ? static_cast<double>(recall3) / validPairs : 0.0)
                      << " recall@6=" << (validPairs > 0 ? static_cast<double>(recall6) / validPairs : 0.0)
                      << " recall@10=" << (validPairs > 0 ? static_cast<double>(recall10) / validPairs : 0.0)
                      << " recall@20=" << (validPairs > 0 ? static_cast<double>(recall20) / validPairs : 0.0)
                      << std::endl;
        }

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "[XFeatVocabEval][ERROR] " << e.what() << std::endl;
        return 1;
    }
}
