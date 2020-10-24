// Sleep-replay model developed in Emergent (www.github.com/emer/emergent)
// Authors: Anna Schapiro, Dhairyya Singh

// Slp-rep runs a hippocampal-cortical model on the Satellite learning task (see Schapiro et al. (2017))
package main

import (
	"flag"
	"fmt"
	"github.com/goki/ki/bitflag"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/schapirolab/leabra-sleep/hip"
	"github.com/schapirolab/leabra-sleep/leabra"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)
func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *leabra.Network   `view:"no-inline"`
	TrainSat     *etable.Table     `view:"no-inline" desc:"training patterns to use"`
	TestSat      *etable.Table     `view:"no-inline" desc:"testing patterns to use"`
	TrnTrlLog    *etable.Table     `view:"no-inline" desc:"training trial-level log data"`
	TrnEpcLog    *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog    *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog    *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstCycLog    *etable.Table     `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog       *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats     *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	TstStats     *etable.Table     `view:"no-inline" desc:"testing stats"`
	Params       params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet     string            `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag          string            `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params)"`
	MaxRuns      int               `desc:"maximum number of model runs to perform"`
	MaxEpcs      int               `desc:"maximum number of epochs to run per model run"`
	NZeroStop    int               `desc:"if a positive number, training will stop after this many epochs with zero mem errors"`
	TrialPerEpc  int               `desc:"number of trials per epoch of training"`
	TrainEnv     env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv      env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time         leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn       bool              `desc:"whether to update the network view while running"`
	TrainUpdt    leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int               `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`

	// Sleep implementation vars
	SleepEnv    env.FixedTable    `desc:"Training environment -- contains everything about iterating over sleep trials"`
	SlpCycLog   *etable.Table     `view:"no-inline" desc:"sleeping cycle-level log data"`
	SlpCycPlot  *eplot.Plot2D     `view:"-" desc:"the sleeping cycle plot"`
	MaxSlpCyc   int               `desc:"maximum number of cycle to sleep for a trial"`
	Sleep       bool              `desc:"Sleep or not"`
	LrnDrgSlp   bool              `desc:"Learning during sleep?"`
	SlpPlusThr  float32           `desc:"The threshold for entering a sleep plus phase"`
	SlpMinusThr float32           `desc:"The threshold for entering a sleep minus phase"`
	InhibOscil  bool              `desc:"whether to implement inhibition oscillation"`
	SleepUpdt   leabra.TimeScales `desc:"at what time scale to update the display during sleep? Anything longer than Epoch updates at Epoch in this model"`
	InhibFactor float64           `desc:"The inhib oscill factor for this cycle"`
	AvgLaySim   float64           `desc:"Average layer similaity between this cycle and last cycle"`
	SynDep      bool              `desc:"Syn Dep during sleep?"`
	SlpLearn    bool              `desc:"Learn during sleep?"`
	PlusPhase   bool              `desc:"Sleep Plusphase on/off"`
	MinusPhase  bool              `desc:"Sleep Minusphase on/off"`
	ZError      int               `desc:"Consec Zero error epochs"`
	ExecSleep	bool			  `desc:"Execute Sleep?"`
	SlpTrls		int				  `desc:"Number of sleep trials"`

	// statistics: note use float64 as that is best for etable.Table - DS Note: TrlSSE, TrlAvgSSE, TrlCosDiff don't need Shared and Unique vals... only accumulators do.
	TestNm     string  `inactive:"+" desc:"what set of patterns are we currently testing"`
	TrlSSE     float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE  float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff float64 `inactive:"+" desc:"current trial's cosine difference"`

	// DS: These accumulators/Epc markers need separate Shared/Unique feature sums for tracking across epcs
	EpcShSSE     float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcShAvgSSE  float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcShPctErr  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcShPctCor  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcShCosDiff float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	ShFirstZero  int     `inactive:"+" desc:"epoch at when Mem err first went to zero"`
	ShNZero      int     `inactive:"+" desc:"number of epochs in a row with zero Mem err"`

	EpcUnSSE     float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcUnAvgSSE  float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcUnPctErr  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcUnPctCor  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcUnCosDiff float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	UnFirstZero  int     `inactive:"+" desc:"epoch at when Mem err first went to zero"`
	UnNZero      int     `inactive:"+" desc:"number of epochs in a row with zero Mem err"`

	// internal state - view:"-"
	// DS: Need separate Shared and Unique feature sums for tracking within epcs
	ShTrlNum     int     `inactive:"+" desc:"last epoch's total number of Shared Trials"`
	ShSumSSE     float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	ShSumAvgSSE  float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	ShSumCosDiff float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	ShCntErr     int     `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`

	UnTrlNum     int     `inactive:"+" desc:"last epoch's total number of Unique Trials"`
	UnSumSSE     float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	UnSumAvgSSE  float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	UnSumCosDiff float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	UnCntErr     int     `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`

	HiddenType    string `view:"-" inactive:"+" desc:"Feature type that is Hidden on this trial - Shared or Unique"`
	HiddenFeature string `view:"-" inactive:"+" desc:"Feature that is Hidden on this trial - F1-F5"`

	Win        *gi.Window       `view:"-" desc:"main GUI window"`
	NetView    *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar    *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	TrnTrlPlot *eplot.Plot2D    `view:"-" desc:"the training trial plot"`
	TrnEpcPlot *eplot.Plot2D    `view:"-" desc:"the training epoch plot"`
	TstEpcPlot *eplot.Plot2D    `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	TstCycPlot *eplot.Plot2D    `view:"-" desc:"the test-cycle plot"`
	RunPlot    *eplot.Plot2D    `view:"-" desc:"the run plot"`
	TrnEpcFile *os.File         `view:"-" desc:"log file"`
	RunFile    *os.File         `view:"-" desc:"log file"`
	TmpVals    []float32        `view:"-" desc:"temp slice for holding values -- prevent mem allocs"`
	LayStatNms []string         `view:"-" desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	TstNms     []string         `view:"-" desc:"names of test tables"`
	SaveWts      bool  `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool  `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool  `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool  `view:"-" desc:"true if sim is running"`
	StopNow      bool  `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool  `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed      int64 `view:"-" desc:"the current random seed"`
	DirSeed      int64 `view:"-" desc:"the current random seed for dir"`

}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.NewRndSeed()
	ss.MaxEpcs = 20
	ss.Net = &leabra.Network{}
	ss.TrainSat = &etable.Table{}
	ss.TestSat = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = SavedParamsSets
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.AlphaCycle
	ss.TestInterval = 1
	ss.LogSetParams = false
	ss.LayStatNms = []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName", "CA1", "DG"}
	ss.TstNms = []string{"Sat"}
	ss.TrialPerEpc = 105
	ss.ShTrlNum = 0
	ss.UnTrlNum = 0
	ss.MaxRuns = 30
	ss.ZError = 0

	ss.SlpCycLog = &etable.Table{}
	ss.Sleep = false
	ss.InhibOscil = true
	ss.SleepUpdt = leabra.Cycle
	ss.MaxSlpCyc = 50000
	ss.SynDep = true
	ss.SlpLearn = true
	ss.PlusPhase = false
	ss.MinusPhase = false
	ss.ExecSleep = true
	ss.SlpTrls = 0
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {

	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)

	ss.ConfigSlpCycLog(ss.SlpCycLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 10
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 50
		ss.NZeroStop = 1
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainSat)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TrainEnv.Trial.Max = ss.TrialPerEpc
	ss.TrainEnv.Sequential = false

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.TestSat)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.SleepEnv.Nm = "SleepEnv"
	ss.SleepEnv.Dsc = "sleep params and state"
	ss.SleepEnv.Table = etable.NewIdxView(ss.TrainSat) // this is needed for the configenv to happen correctly even if no pats are ever shown
	ss.SleepEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
	ss.SleepEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "sleep-replay")

	// Higer-level visual areas
	feature1 := net.AddLayer2D("F1", 6, 1, emer.Input)
	feature2 := net.AddLayer2D("F2", 6, 1, emer.Input)
	feature3 := net.AddLayer2D("F3", 6, 1, emer.Input)
	feature4 := net.AddLayer2D("F4", 6, 1, emer.Input)
	feature5 := net.AddLayer2D("F5", 6, 1, emer.Input)

	// Higer-level language areas
	classname := net.AddLayer2D("ClassName", 1, 3, emer.Input)
	codename := net.AddLayer2D("CodeName", 6, 15, emer.Input)

	// Hipocampus!
	dg := net.AddLayer2D("DG", 15, 15, emer.Hidden)
	ca3 := net.AddLayer2D("CA3", 12, 12, emer.Hidden)
	ca1 := net.AddLayer2D("CA1", 10, 10, emer.Hidden)

	feature1.SetClass("Per")
	feature2.SetClass("Per")
	feature3.SetClass("Per")
	feature4.SetClass("Per")
	feature5.SetClass("Per")

	classname.SetClass("Per")
	codename.SetClass("Per")

	dg.SetClass("Hip")
	ca1.SetClass("Hip")
	ca3.SetClass("Hip")

	feature1.SetPos(mat32.Vec3{0, 20, 0})
	feature2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F1", YAlign: relpos.Front, Space: 2})
	feature3.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F2", YAlign: relpos.Front, Space: 2})
	feature4.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F3", YAlign: relpos.Front, Space: 2})
	feature5.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F4", YAlign: relpos.Front, Space: 2})
	codename.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F5", YAlign: relpos.Front, Space: 2})
	classname.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "CodeName", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "F1", YAlign: relpos.Front, Space: 5})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "DG", YAlign: relpos.Front, Space: 2})
	ca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "DG", YAlign: relpos.Front, Space: 5})

	PerLays := []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName"}

	conn := prjn.NewFull()

	spconn2 := prjn.NewUnifRnd()
	spconn2.PCon = 0.05
	spconn2.RndSeed = ss.RndSeed

	// Per-Hip
	for _, lyc := range PerLays {

		spconn := prjn.NewUnifRnd()
		spconn.PCon = 0.09 // 0.09 is the limit for how sparse you can get here.
		spconn.RndSeed = ss.RndSeed

		ly := ss.Net.LayerByName(lyc).(leabra.LeabraLayer).AsLeabra()

		pj := net.ConnectLayersPrjn(ly, dg, spconn, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("PerDGPrjn")

		pj = net.ConnectLayersPrjn(ly, ca3, spconn, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("PerDGPrjn")

		pj = net.ConnectLayersPrjn(ly, ca1, conn, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("PerCA1Prjn")
		pj = net.ConnectLayersPrjn(ca1, ly, conn, emer.Back, &hip.CHLPrjn{})
		pj.SetClass("PerCA1Prjn")

		time.Sleep(1)
		ss.NewRndSeed()
	}

	pj := net.ConnectLayersPrjn(ca3, ca3, conn, emer.Lateral, &hip.CHLPrjn{})
	pj.SetClass("HipPrjn")

	pj = net.ConnectLayersPrjn(dg, ca3, spconn2, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HipPrjn")

	pj = net.ConnectLayersPrjn(ca3, ca1, conn, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HipPrjn")

	pj = net.ConnectLayersPrjn(codename, codename, conn, emer.Lateral, &hip.CHLPrjn{})
	pj.SetClass("CodePrjn")

	//using 3 threads :)
	dg.SetThread(1)
	ca1.SetThread(2)
	ca3.SetThread(3)

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// outLay.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.NewRun()
	ss.UpdateView("train")
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
	//fmt.Println(ss.RndSeed)
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(state string) string { // changed from boolean to string
	if state == "train" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tTrial:"+" "+"%d\tCycle:"+" "+"%d\tName:"+" "+"%v\t\tHidden:"+" "+"%v\tFeature:"+" "+"%s\t\t\t\nShared Percent Correct:"+" "+"%.2f\t Unique Percent Correct"+" "+"%.2f\tUnique SSE"+" "+"%.2f\tShared SSE"+" "+"%.2f\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, fmt.Sprintf(ss.TrainEnv.TrialName.Cur), ss.HiddenType, ss.HiddenFeature, ss.EpcShPctCor, ss.EpcUnPctCor, ss.EpcUnSSE, ss.EpcShSSE)
	} else if state == "test" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tTrial:"+" "+"%d\tCycle:"+" "+"%d\tName:"+" "+"%v\t\tHidden:"+" "+"%v\tFeature:"+" "+"%s\t\t\t\nShared Percent Correct:"+" "+"%.2f\t Unique Percent Correct"+" "+"%.2f\tUnique SSE"+" "+"%.2f\tShared SSE"+" "+"%.2f\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, fmt.Sprintf(ss.TestEnv.TrialName.Cur), ss.HiddenType, ss.HiddenFeature, ss.EpcShPctCor, ss.EpcUnPctCor, ss.EpcUnSSE, ss.EpcShSSE)
	} else if state == "sleep" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tCycle:"+" "+"%d\tInhibFactor: "+" "+"%.6f\tAvgLaySim: "+" "+"%.6f\t\t\t\nShared Percent Correct:"+" "+"%.2f\t Unique Percent Correct:"+" "+"%.2f\t PlusPhase:"+" "+"%t\t MinusPhase:"+" "+"%t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.Time.Cycle, ss.InhibFactor, ss.AvgLaySim, ss.EpcShPctCor, ss.EpcUnPctCor, ss.PlusPhase, ss.MinusPhase)
	}
	return ""
}

func (ss *Sim) UpdateView(state string) { // changed from boolean to string
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(state))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

func (ss *Sim) SleepCycInit() {

	ss.Time.Reset()

	// Set all layers to be hidden
	for _, ly := range ss.Net.Layers {
		ly.SetType(emer.Hidden)
	}
	ss.Net.InitActs()

	// Set all layers to random activation
	for _, ly := range ss.Net.Layers {
		for ni := range ly.(*leabra.Layer).Neurons {
			nrn := &ly.(*leabra.Layer).Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			msk := bitflag.Mask32(int(leabra.NeurHasExt))
			nrn.ClearMask(msk)
			rnd := rand.Float32()
			rnd = rnd-0.5
			if rnd < 0 {
				rnd = 0
			}
			nrn.Act = rnd

		}
		ss.UpdateView("sleep")
	}

	if ss.SynDep {
		for _, ly := range
		ss.Net.Layers {
			inc := 0.0007 // 0.0015
			dec := 0.0005 // 0.0005
			ly.(*leabra.Layer).InitSdEffWt(float32(inc), float32(dec))
		}
	}


}

func (ss *Sim) BackToWake() {
	// Effwt back to =Wt
	if ss.SynDep {
		for _, ly := range ss.Net.Layers {
			ly.(*leabra.Layer).TermSdEffWt()
		}
	}

	// Set the input/output/hidden layers back to normal.
	iolynms := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
	for _, lynm := range iolynms {
		ly := ss.Net.LayerByName(lynm).(leabra.LeabraLayer).AsLeabra()
		ly.SetType(emer.Input)
		ly.UpdateExtFlags()
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()

	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt()
	}

	if train {
		perlys := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
		for _, ly := range perlys {
			lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
			lycToca1 := lyc.SndPrjns.RecvName("CA1").(*hip.CHLPrjn)
			lycToca1.WtScale.Abs = 1
		}
		ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn).WtScale.Abs = 0
	}
	if !train {
		perlys := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
		for _, ly := range perlys {
			lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
			lycToca1 := lyc.SndPrjns.RecvName("CA1").(*hip.CHLPrjn)
			lycToca1.WtScale.Abs = 1
		}
		ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn).WtScale.Abs = 1
	}

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < 25; cyc++ {
			ss.Net.Cycle(&ss.Time, false)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					ss.UpdateView("train")
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView("train")
					}
				}
			}

		}
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			if train {
				perlys := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
				for _, ly := range perlys {
					lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
					lycToca1 := lyc.SndPrjns.RecvName("CA1").(*hip.CHLPrjn)
					lycToca1.WtScale.Abs = 0
				}
				ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn).WtScale.Abs = 1
			}
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

		case 3: // Fourth Quarter: CA1 back to ECin drive only
			if train {
				perlys := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
				for _, ly := range perlys {
					lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
					lycToca1 := lyc.SndPrjns.RecvName("CA1").(*hip.CHLPrjn)
					lycToca1.WtScale.Abs = 1
				}
				ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn).WtScale.Abs = 0
			}
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

		}
		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			//ss.MemStats(train) // must come after QuarterFinal DS: Deprecated for sleep-replay
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView("train")
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView("train")
				}
			}
		}
	}

	perlys := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
	for _, ly := range perlys {
		lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
		lycTodg := lyc.SndPrjns.RecvName("DG").(*hip.CHLPrjn)
		lycTodg.WtScale.Abs = 1
		lycToca1 := lyc.SndPrjns.RecvName("CA1").(*hip.CHLPrjn)
		lycToca1.WtScale.Abs = 1
	}
	ca1.RcvPrjns.SendName("CA3").(*hip.CHLPrjn).WtScale.Abs = 1

	if ss.TrainEnv.Run.Cur == 0 {
		ss.DirSeed = ss.RndSeed
	}

	if train {
		ss.Net.DWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView("train")
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}

	// DS: Sleep check needs to be on top because criterion stats only get computed at the end of the epoch
	// and if check is at the end, one extra trn trial will hapen before sleep

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView("train")
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()

			if ss.EpcShPctCor >= 0.8 && ss.EpcUnPctCor >= 0.8{

				if ss.ExecSleep{


					fmt.Println([]string{strconv.FormatFloat(ss.EpcShPctCor , 'f', 6, 64), strconv.FormatFloat(ss.EpcUnPctCor , 'f', 6, 64), strconv.FormatFloat(ss.EpcShSSE , 'f', 6, 64), strconv.FormatFloat(ss.EpcUnSSE , 'f', 6, 64)})
					ss.SleepTrial()
					fmt.Println(ss.EpcShPctCor, ss.EpcUnPctCor, ss.EpcShSSE, ss.EpcUnSSE)
					ss.TestAll()
					fmt.Println(ss.EpcShPctCor, ss.EpcUnPctCor, ss.EpcShSSE, ss.EpcUnSSE)
				}

				ss.RunEnd()
				if ss.TrainEnv.Run.Incr() { // we are done!
					ss.StopNow = true
					return
				} else {
					ss.NeedsNewRun = true
					return
				}
			}
		}
		learned := (ss.NZeroStop > 0 && ss.ShNZero >= ss.NZeroStop && ss.UnNZero >= ss.NZeroStop)

		if learned || epc >= ss.MaxEpcs { // done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	// Setting up train trial layer input/target chnages in this block
	f1 := ss.Net.LayerByName("F1").(leabra.LeabraLayer).AsLeabra()
	f2 := ss.Net.LayerByName("F2").(leabra.LeabraLayer).AsLeabra()
	f3 := ss.Net.LayerByName("F3").(leabra.LeabraLayer).AsLeabra()
	f4 := ss.Net.LayerByName("F4").(leabra.LeabraLayer).AsLeabra()
	f5 := ss.Net.LayerByName("F5").(leabra.LeabraLayer).AsLeabra()
	classname := ss.Net.LayerByName("ClassName").(leabra.LeabraLayer).AsLeabra()
	codename := ss.Net.LayerByName("CodeName").(leabra.LeabraLayer).AsLeabra()

	name := (ss.TrainEnv.TrialName.Cur)
	unique := 0
	shared := []string{"1", "2", "3", "4", "5", "classname"}
	r := rand.Float64()
	r1 := rand.Float64()
	outlay := ""

	for i, j := range name {
		if string(j) == "4" || string(j) == "5" || string(j) == "6" {
			unique = i  + 1
			break
		}
	}

	for i, v := range shared {
		if (v) == strconv.Itoa(unique) {
			shared = append(shared[:i], shared[i+1:]...)
			break
		}
	}

	//Setting ratio for shared:unique feature hiding - fix code here later
	if r > 0.95 { // shared
		ss.HiddenType = "shared"
		hideindex := int(rand.Intn(len(shared)))
		ss.HiddenFeature = (shared[hideindex])
		ss.ShTrlNum++
	} else { // unique
		if unique == 0 { // if there are no unique features, set codename to hide
			ss.HiddenType = "unique"
			ss.HiddenFeature = "codename"
			ss.UnTrlNum++
		} else {
			ss.HiddenType = "unique"
			ss.UnTrlNum++
			if r1 > 0.5 {
				ss.HiddenFeature = strconv.Itoa(unique)
			} else {
				ss.HiddenFeature = "codename"
			}
		}

	}

	// Using the feature number that has to be atl to change the right feature layer to emer.Target
	switch ss.HiddenFeature {
	case "1":
		f1.SetType(emer.Target)
		f1.UpdateExtFlags()
		outlay = f1.Name()
	case "2":
		f2.SetType(emer.Target)
		f2.UpdateExtFlags()
		outlay = f2.Name()
	case "3":
		f3.SetType(emer.Target)
		f3.UpdateExtFlags()
		outlay = f3.Name()
	case "4":
		f4.SetType(emer.Target)
		f4.UpdateExtFlags()
		outlay = f4.Name()
	case "5":
		f5.SetType(emer.Target)
		f5.UpdateExtFlags()
		outlay = f5.Name()
	case "classname":
		classname.SetType(emer.Target)
		classname.UpdateExtFlags()
		outlay = classname.Name()
	case "codename":
		codename.SetType(emer.Target)
		codename.UpdateExtFlags()
		outlay = codename.Name()
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true) // train

	ss.TrialStats(true, outlay) // accumulate

	f1.SetType(emer.Input)
	f1.UpdateExtFlags()
	f2.SetType(emer.Input)
	f2.UpdateExtFlags()
	f3.SetType(emer.Input)
	f3.UpdateExtFlags()
	f4.SetType(emer.Input)
	f4.UpdateExtFlags()
	f5.SetType(emer.Input)
	f5.UpdateExtFlags()
	classname.SetType(emer.Input)
	classname.UpdateExtFlags()
	codename.SetType(emer.Input)
	codename.UpdateExtFlags()

	ss.LogTrnTrl(ss.TrnTrlLog)
}


func (ss *Sim) SleepCyc(c [][]float64) {

	viewUpdt := ss.SleepUpdt
	//ss.Time.SleepCycStart()

	ss.Net.WtFmDWt()

	stablecount := 0
	pluscount := 0
	minuscount := 0
	ss.SlpTrls = 0

	finhib := ss.Net.LayerByName("F1").(*leabra.Layer).Inhib.Layer.Gi
	clinhib := ss.Net.LayerByName("ClassName").(*leabra.Layer).Inhib.Layer.Gi
	coinhib := ss.Net.LayerByName("CodeName").(*leabra.Layer).Inhib.Layer.Gi
	dginhib := ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi
	ca1inhib := ss.Net.LayerByName("CA1").(*leabra.Layer).Inhib.Layer.Gi
	ca3inhib := ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi


	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	ca3.RcvPrjns.SendName("CA3").(*hip.CHLPrjn).WtScale.Abs = 2

	perlys := []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName" }
	for _, ly := range perlys {
		lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
		lycfmca1 := lyc.RcvPrjns.SendName("CA1").(*hip.CHLPrjn)
		lycfmca1.WtScale.Abs = 2
	}

	ss.Net.GScaleFmAvgAct() // update computed scaling factors
	ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

	for cyc := 0; cyc < 30000; cyc++ {

		ss.Net.WtFmDWt()

		ss.Net.Cycle(&ss.Time, true)
		ss.UpdateView("sleep")

		// Taking the prepared slice of oscil inhib values and producing the oscils in all perlys
		if ss.InhibOscil {
			inhibs := c
			ss.InhibFactor = inhibs[0][cyc] // For sleep GUI counter and sleepcyclog

			// Changing Inhibs back to default before next oscill cycle value so that the inhib values follow a sinwave
			perlys := []string{"F1", "F2", "F3", "F4", "F5"}
			for _, ly := range perlys {
				ss.Net.LayerByName(ly).(*leabra.Layer).Inhib.Layer.Gi = finhib
			}
			ss.Net.LayerByName("ClassName").(*leabra.Layer).Inhib.Layer.Gi = clinhib
			ss.Net.LayerByName("CodeName").(*leabra.Layer).Inhib.Layer.Gi = coinhib
			ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi = dginhib
			ss.Net.LayerByName("CA1").(*leabra.Layer).Inhib.Layer.Gi = ca1inhib
			ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi = ca3inhib

			lowlayers := []string{ "ClassName", "CA1", "CodeName"}
			highlayers := []string{ "F1", "F2", "F3", "F4", "F5", "DG", "CA3"}

			for _, layer := range lowlayers {
				ly := ss.Net.LayerByName(layer).(*leabra.Layer)
				ly.Inhib.Layer.Gi = ly.Inhib.Layer.Gi * float32(inhibs[0][cyc])
			}
			for _, layer := range highlayers {
				ly := ss.Net.LayerByName(layer).(*leabra.Layer)
				ly.Inhib.Layer.Gi = ly.Inhib.Layer.Gi * float32(inhibs[1][cyc])
			}
		}

		//DS: average network similarity
		avesim := 0.0
		tmpsim := 0.0
		for _, lyc := range ss.Net.Layers {
			ly := ss.Net.LayerByName(lyc.Name()).(*leabra.Layer)
			tmpsim = ly.Sim
			//fmt.Println(ly.Name(), tmpsim)
			if math.IsNaN(tmpsim) {
				tmpsim = 0
			}
			avesim = avesim + tmpsim
		}
		ss.AvgLaySim = avesim / 10

		//If AvgLaySim falls below 0.9 - most likely because a layer has lost all act, random noise will be injected
		//into the network to get it going again. The first 1000 cycles are skipped to let the network initially settle into an attractor.
		if ss.Time.Cycle > 200 && ss.AvgLaySim <= 0.8  && (ss.Time.Cycle % 50 == 0 || ss.Time.Cycle % 50 == 1 || ss.Time.Cycle % 50 == 2 || ss.Time.Cycle % 50 == 3 || ss.Time.Cycle % 50 == 4) {
			for _, ly := range ss.Net.Layers {
				for ni := range ly.(*leabra.Layer).Neurons {
					nrn := &ly.(*leabra.Layer).Neurons[ni]
					if nrn.IsOff() {
						continue
					}
					nrn.Act = 0
					rnd := rand.Float32()
					rnd = rnd-0.5
					if rnd < 0 {
						rnd = 0
					}
					nrn.Act = rnd
				}
			}
		}

		// Logging the SlpCycLog
		ss.LogSlpCyc(ss.SlpCycLog, ss.Time.Cycle)

		// Mark plus or minus phase
		if ss.SlpLearn {

			plusthresh := 0.9999938129217251 + 0.0000055
			minusthresh := 0.9999938129217251 - 0.001

			// Checking if stable
			if ss.PlusPhase == false && ss.MinusPhase == false {
				if ss.AvgLaySim >= plusthresh {
					stablecount++
				} else if ss.AvgLaySim < plusthresh {
					stablecount = 0
				}
			}

			// For a dual threshold model, checking here if network has been stable above plusthresh for 5 cycles
			// Starting plus phase if criteria met
			if stablecount == 5 && ss.AvgLaySim >= plusthresh && ss.PlusPhase == false && ss.MinusPhase == false { //&& ss.Time.Quarter == 0 {
				stablecount = 0
				minuscount = 0
				ss.PlusPhase = true
				pluscount++
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(true)
				}
				fmt.Println(cyc, "plusphase begins")

			} else if pluscount > 0 && ss.AvgLaySim >= plusthresh && ss.PlusPhase == true {
				pluscount++
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(false)
				}
			} else if ss.AvgLaySim < plusthresh && ss.AvgLaySim >= minusthresh && ss.PlusPhase == true {
				ss.PlusPhase = false
				ss.MinusPhase = true
				minuscount++

				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().CalcActP(pluscount)
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(true)
				}
				pluscount = 0
				fmt.Println(cyc, "plusphase ends, minusphase begins")

			} else if ss.AvgLaySim >= minusthresh && ss.MinusPhase == true {
				minuscount++
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(false)
				}
			} else if ss.AvgLaySim < minusthresh && ss.MinusPhase == true {
				ss.MinusPhase = false

				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().CalcActM(minuscount)
				}
				minuscount = 0
				stablecount = 0

				for _, lyc := range ss.Net.Layers {
					ss.SlpTrls++
					ly := ss.Net.LayerByName(lyc.Name()).(*leabra.Layer)
					for _, p := range ly.SndPrjns {
						if p.IsOff() {
							continue
						}
						p.(*hip.CHLPrjn).SlpDWt()
					}
				}
				fmt.Println(cyc, "minusphase ends; Dwt occured")
			} else if ss.AvgLaySim < minusthresh && ss.PlusPhase == true {
				ss.PlusPhase = false
				pluscount = 0
				stablecount = 0
				minuscount = 0
			}
		}

		// Forward the cycle timer
		ss.Time.CycleInc()

		ss.UpdateView("sleep")
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Cycle:
				ss.UpdateView("sleep")
			case leabra.FastSpike:
				if (cyc+1)%10 == 0 {
					ss.UpdateView("sleep")
				}
			case leabra.Quarter:
				if (cyc+1)%25 == 0 {
					ss.UpdateView("sleep")
				}
			case leabra.Phase:
				if (cyc+1)%100 == 0 {
					ss.UpdateView("sleep")
				}
			}
		}
	}

	pluscount = 0
	minuscount = 0
	ss.MinusPhase = false
	ss.PlusPhase = false
	stablecount = 0

	perlys = []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
	for _, ly := range perlys {
		lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
		lycfmca1 := lyc.RcvPrjns.SendName("CA1").(*hip.CHLPrjn)
		lycfmca1.WtScale.Abs = 1
		ca1fmlyc := lyc.SndPrjns.RecvName("CA1").(*hip.CHLPrjn)
		ca1fmlyc.WtScale.Abs = 1
		dgfmlyc :=  lyc.SndPrjns.RecvName("DG").(*hip.CHLPrjn)
		dgfmlyc.WtScale.Abs = 1
	}
	ca3.RcvPrjns.SendName("CA3").(*hip.CHLPrjn).WtScale.Abs = 1

	ss.Net.GScaleFmAvgAct() // update computed scaling factors
	ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

	perlys = []string{"F1", "F2", "F3", "F4", "F5"}
	for _, ly := range perlys {
		ss.Net.LayerByName(ly).(*leabra.Layer).Inhib.Layer.Gi = finhib
	}

	ss.Net.LayerByName("ClassName").(*leabra.Layer).Inhib.Layer.Gi = clinhib
	ss.Net.LayerByName("CodeName").(*leabra.Layer).Inhib.Layer.Gi = coinhib
	ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi = dginhib
	ss.Net.LayerByName("CA1").(*leabra.Layer).Inhib.Layer.Gi = ca1inhib

	if ss.ViewOn {
		ss.UpdateView("sleep")
	}
}

// DZ added for sleep
func (ss *Sim) SleepTrial() {
	ss.SleepCycInit()
	ss.UpdateView("sleep")

	// DS added for inhib oscill
	start := 0.
	stop := 10000.
	step := 0.1

	N := int(math.Ceil((stop - start) / step))
	rnge := make([]float64, N)
	for x := range rnge {
		rnge[x] = start + step*float64(x)
	}
	a := rnge
	c := make([][]float64, 2)
	for i := 0; i < 100000; i++ {
		c[0] = append(c[0], (math.Cos(a[i]/1)/80 + 0.99))
		c[1] = append(c[1], (math.Cos(a[i]/1)/30 + 0.99))
	}
	ss.SleepCyc(c)
	ss.SlpCycPlot.GoUpdate()
	ss.BackToWake()
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.NewRndSeed()
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainSat)
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()

	ss.InitStats()
	ss.TrnTrlLog.SetNumRows(0)
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false

	dg := ss.Net.LayerByName("DG").(*leabra.Layer)
	ca3 := ss.Net.LayerByName("CA3").(*leabra.Layer)

	pjdgca3 := ca3.RcvPrjns.SendName("DG").(*hip.CHLPrjn)
	pjdgca3.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
	pjdgca3.Build()

	perlys := []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName"}
	for _, layer := range perlys {
		time.Sleep(1)
		ss.NewRndSeed()

		pjperca3 := ca3.RcvPrjns.SendName(layer).(*hip.CHLPrjn)
		pjperca3.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
		pjperca3.Build()

		//pjca3per := ca3.SndPrjns.RecvName(layer).(*hip.CHLPrjn)
		//pjca3per.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
		//pjca3per.Build()

		pjperdg := dg.RcvPrjns.SendName(layer).(*hip.CHLPrjn)
		pjperdg.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
		pjperdg.Build()
	}

	ss.Net.InitWts()


	ss.TrainEnv.Trial.Max = ss.TrialPerEpc // DS added

	fmt.Println(ss.TrainEnv.Run.Cur)

}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators for Shared Trials
	ss.ShSumSSE = 0
	ss.ShSumAvgSSE = 0
	ss.ShSumCosDiff = 0
	ss.ShCntErr = 0
	ss.ShFirstZero = -1
	ss.ShNZero = 0

	// accumulators for Unique Trials
	ss.UnSumSSE = 0
	ss.UnSumAvgSSE = 0
	ss.UnSumCosDiff = 0
	ss.UnCntErr = 0
	ss.UnFirstZero = -1
	ss.UnNZero = 0

	// epc tracking of shared/unique feature accums
	ss.EpcShSSE = 0
	ss.EpcShAvgSSE = 0
	ss.EpcShPctErr = 0
	ss.EpcShCosDiff = 0

	ss.EpcUnSSE = 0
	ss.EpcUnAvgSSE = 0
	ss.EpcUnPctErr = 0
	ss.EpcUnCosDiff = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool, outlaynm string) (sse, avgsse, cosdiff float64) {

	outLay := ss.Net.LayerByName(outlaynm).(leabra.LeabraLayer).AsLeabra()

	// CosDiff calculates the cosine diff between ActM and ActP
	// MSE calculates the sum squared error and the mean squared error for the OutLay
	ss.TrlCosDiff = float64(outLay.CosDiff.Cos)
	ss.TrlSSE, ss.TrlAvgSSE = outLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if accum {
		if ss.HiddenType == "shared" {
			ss.ShSumSSE += ss.TrlSSE
			ss.ShSumAvgSSE += ss.TrlAvgSSE
			ss.ShSumCosDiff += ss.TrlCosDiff
			if ss.TrlSSE != 0 {
				ss.ShCntErr++
			}
		}
		if ss.HiddenType == "unique" {
			ss.UnSumSSE += ss.TrlSSE
			ss.UnSumAvgSSE += ss.TrlAvgSSE
			ss.UnSumCosDiff += ss.TrlCosDiff
			if ss.TrlSSE != 0 {
				ss.UnCntErr++
			}
		}
	}
	return
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	curTrial := ss.TrainEnv.Trial.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc || curTrial == ss.TrialPerEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		vp.BlockUpdates()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.UnblockUpdates()
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView("test")
		}
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false) // !train

	// Setting up train trial layer input/target chnages in this block
	f1 := ss.Net.LayerByName("F1").(leabra.LeabraLayer).AsLeabra()
	f2 := ss.Net.LayerByName("F2").(leabra.LeabraLayer).AsLeabra()
	f3 := ss.Net.LayerByName("F3").(leabra.LeabraLayer).AsLeabra()
	f4 := ss.Net.LayerByName("F4").(leabra.LeabraLayer).AsLeabra()
	f5 := ss.Net.LayerByName("F5").(leabra.LeabraLayer).AsLeabra()
	classname := ss.Net.LayerByName("ClassName").(leabra.LeabraLayer).AsLeabra()
	codename := ss.Net.LayerByName("CodeName").(leabra.LeabraLayer).AsLeabra()

	outlay := ""

	switch ss.HiddenFeature {
	case "1":
		outlay = f1.Name()
	case "2":
		outlay = f2.Name()
	case "3":
		outlay = f3.Name()
	case "4":
		outlay = f4.Name()
	case "5":
		outlay = f5.Name()
	case "classname":
		outlay = classname.Name()
	case "codename":
		outlay = codename.Name()
	}
	ss.TrialStats(true, outlay) // !accumulate
}

// TestItem tests given item which is at given index in test item list
// Currently Testitem will not do trialstats accum
func (ss *Sim) TestItem(idx int) {

	//outlay := ""
	//hide := ""

	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false) // !train
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	//fmt.Println(ss.TestEnv.TrialName)

	ss.TestNm = "Train Sat Permutations"
	ss.TestEnv.Table = etable.NewIdxView(ss.TestSat)
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)

	ss.HiddenType = ""
	ss.HiddenFeature = ""
	ss.UnTrlNum = 0
	ss.ShTrlNum = 0

	// Setting up train trial layer input/target chnages in this block
			f1 := ss.Net.LayerByName("F1").(leabra.LeabraLayer).AsLeabra()
			f2 := ss.Net.LayerByName("F2").(leabra.LeabraLayer).AsLeabra()
			f3 := ss.Net.LayerByName("F3").(leabra.LeabraLayer).AsLeabra()
			f4 := ss.Net.LayerByName("F4").(leabra.LeabraLayer).AsLeabra()
			f5 := ss.Net.LayerByName("F5").(leabra.LeabraLayer).AsLeabra()
			classname := ss.Net.LayerByName("ClassName").(leabra.LeabraLayer).AsLeabra()
			codename := ss.Net.LayerByName("CodeName").(leabra.LeabraLayer).AsLeabra()

			for i := 0; i < 7; i++ {
				for j := 0; j < 15; j++ {

					switch i {
					case 0:
						f1.SetType(emer.Target)
						f1.UpdateExtFlags()
						ss.HiddenFeature = "1"
					case 1:
						f2.SetType(emer.Target)
						f2.UpdateExtFlags()
						ss.HiddenFeature = "2"
					case 2:
						f3.SetType(emer.Target)
						f3.UpdateExtFlags()
						ss.HiddenFeature = "3"
					case 3:
						f4.SetType(emer.Target)
						f4.UpdateExtFlags()
						ss.HiddenFeature = "4"
					case 4:
						f5.SetType(emer.Target)
						f5.UpdateExtFlags()
						ss.HiddenFeature = "5"
					case 5:
						classname.SetType(emer.Target)
						classname.UpdateExtFlags()
						ss.HiddenFeature = "classname"
					case 6:
						codename.SetType(emer.Target)
						codename.UpdateExtFlags()
						ss.HiddenFeature = "codename"
					}

					ss.TestTrial(true) // return on chg

					name := ss.TestEnv.TrialName.Cur

					ss.HiddenType = "shared"
					for n, feature := range name {
						if (i < 5) && (string(feature) == "4" || string(feature) == "5" || string(feature) == "6") && (n == i) { // checking here if there is a unique feature and if it is the currently hidden one
							ss.HiddenType = "unique"
							break
						} else if i == 6 {
							ss.HiddenType = "unique"
							break
						}
					}

			ss.LogTstTrl(ss.TstTrlLog)

			f1.SetType(emer.Input)
			f1.UpdateExtFlags()
			f2.SetType(emer.Input)
			f2.UpdateExtFlags()
			f3.SetType(emer.Input)
			f3.UpdateExtFlags()
			f4.SetType(emer.Input)
			f4.UpdateExtFlags()
			f5.SetType(emer.Input)
			f5.UpdateExtFlags()
			classname.SetType(emer.Input)
			classname.UpdateExtFlags()
			codename.SetType(emer.Input)
			codename.UpdateExtFlags()

			_, _, chg := ss.TestEnv.Counter(env.Epoch)
			if chg || ss.StopNow {
				break
			}

			if ss.HiddenType == "unique" {
				ss.UnTrlNum++
			} else {
				ss.ShTrlNum++
			}
		}
	}


	// log only at very end
	ss.LogTstEpc(ss.TstEpcLog)

}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		err = ss.SetParamsSet(ss.ParamSet, sheet, setMsg)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) OpenPat(dt *etable.Table, fname, name, desc string) {
	err := dt.OpenCSV(gi.FileName(fname), etable.Tab)
	if err != nil {
		log.Println(err)
		return
	}
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
}

func (ss *Sim) OpenPats() {
	ss.OpenPat(ss.TrainSat, "Train_Sats_go.txt", "TrainSat", "Training Patterns")
	ss.OpenPat(ss.TestSat, "Test_Sats_go.txt", "TestSat", "Testing Patterns")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur

	row := dt.Rows
	if trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, (ss.TrainEnv.TrialName.Cur))
	dt.SetCellString("HiddenType", row, ss.HiddenType)
	dt.SetCellString("HiddenFeature", row, (ss.HiddenFeature))
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {

	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"HiddenType", etensor.STRING, nil, nil},
		{"HiddenFeature", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Train Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("HiddenType", true, true, 0, false, 0)
	plt.SetColParams("HiddenFeature", false, true, 0, false, 0)
	plt.SetColParams("SSE", true, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	return plt
}

func (ss *Sim) LogSlpCyc(dt *etable.Table, cyc int) {

	row := dt.Rows
	if cyc == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	dt.SetCellFloat("InhibFactor", cyc, float64(ss.InhibFactor))
	dt.SetCellFloat("AvgLaySim", cyc, float64(ss.AvgLaySim))

	for _, ly := range ss.Net.Layers {
		lyc := ss.Net.LayerByName(ly.Name()).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Name()+" Sim", row, float64(lyc.Sim))
	}

	ss.SlpCycPlot.GoUpdate()

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
	}
}

//DZ added
func (ss *Sim) ConfigSlpCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "SlpCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one sleep trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := ss.MaxSlpCyc // max cycles

	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
		{"InhibFactor", etensor.FLOAT64, nil, nil},
		{"AvgLaySim", etensor.FLOAT64, nil, nil},
	}

	for _, ly := range ss.Net.Layers {
		sch = append(sch, etable.Column{ly.Name() + " Sim", etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, np)
}

//DZ added
func (ss *Sim) ConfigSlpCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Sleep Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", true, true, 0, false, 0)
	plt.SetColParams("AvgLaySim", true, true, 0, true, 1)
	return plt
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv         // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Trial.Max) // number of trials in view
	shnt := float64(ss.ShTrlNum)
	unnt := float64(ss.UnTrlNum)

	// Computing Epc Shared/Unique feature learning metrics
	ss.EpcShSSE = ss.ShSumSSE / shnt
	ss.ShSumSSE = 0
	ss.EpcShAvgSSE = ss.ShSumAvgSSE / shnt
	ss.ShSumAvgSSE = 0
	ss.EpcShPctErr = float64(ss.ShCntErr) / shnt
	ss.ShCntErr = 0
	ss.EpcShPctCor = 1 - ss.EpcShPctErr
	ss.EpcShCosDiff = ss.ShSumCosDiff / shnt
	ss.ShSumCosDiff = 0
	ss.ShTrlNum = 0

	ss.EpcUnSSE = ss.UnSumSSE / unnt
	ss.UnSumSSE = 0
	ss.EpcUnAvgSSE = ss.UnSumAvgSSE / unnt
	ss.UnSumAvgSSE = 0
	ss.EpcUnPctErr = float64(ss.UnCntErr) / unnt
	ss.UnCntErr = 0
	ss.EpcUnPctCor = 1 - ss.EpcUnPctErr
	ss.EpcUnCosDiff = ss.UnSumCosDiff / unnt
	ss.UnSumCosDiff = 0
	ss.UnTrlNum = 0


	// Adding shared/unique metrics to log
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Total Trials", row, float64(nt))
	dt.SetCellFloat("Shared Trials", row, float64(shnt))
	dt.SetCellFloat("ShSSE", row, ss.EpcShSSE)
	dt.SetCellFloat("ShAvgSSE", row, ss.EpcShAvgSSE)
	dt.SetCellFloat("ShPctErr", row, ss.EpcShPctErr)
	dt.SetCellFloat("ShPctCor", row, ss.EpcShPctCor)
	dt.SetCellFloat("ShCosDiff", row, ss.EpcShCosDiff)
	dt.SetCellFloat("Unique Trials", row, float64(unnt))
	dt.SetCellFloat("UnSSE", row, ss.EpcUnSSE)
	dt.SetCellFloat("UnAvgSSE", row, ss.EpcUnAvgSSE)
	dt.SetCellFloat("UnPctErr", row, ss.EpcUnPctErr)
	dt.SetCellFloat("UnPctCor", row, ss.EpcUnPctCor)
	dt.SetCellFloat("UnCosDiff", row, ss.EpcUnCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()

	if ss.EpcUnSSE == 0 && ss.EpcShSSE == 0 {
		ss.ZError++
	} else {
		ss.ZError = 0
	}

}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Total Trials", etensor.INT64, nil, nil},
		{"Shared Trials", etensor.INT64, nil, nil},
		{"ShSSE", etensor.FLOAT64, nil, nil},
		{"ShAvgSSE", etensor.FLOAT64, nil, nil},
		{"ShPctErr", etensor.FLOAT64, nil, nil},
		{"ShPctCor", etensor.FLOAT64, nil, nil},
		{"ShCosDiff", etensor.FLOAT64, nil, nil},
		{"Unique Trials", etensor.INT64, nil, nil},
		{"UnSSE", etensor.FLOAT64, nil, nil},
		{"UnAvgSSE", etensor.FLOAT64, nil, nil},
		{"UnPctErr", etensor.FLOAT64, nil, nil},
		{"UnPctCor", etensor.FLOAT64, nil, nil},
		{"UnCosDiff", etensor.FLOAT64, nil, nil},

		//{"Mem", etensor.FLOAT64, nil, nil},
		//{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		//{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("ShSSE", true, true, 0, false, 0)
	plt.SetColParams("ShAvgSSE", false, true, 0, false, 0)
	plt.SetColParams("ShPctErr", false, true, 0, true, 1)
	plt.SetColParams("ShPctCor", true, true, 0, true, 1)
	plt.SetColParams("ShCosDiff", false, true, 0, true, 1)
	plt.SetColParams("UnSSE", true, true, 0, false, 0)
	plt.SetColParams("UnAvgSSE", false, true, 0, false, 0)
	plt.SetColParams("UnPctErr", false, true, 0, true, 1)
	plt.SetColParams("UnPctCor", true, true, 0, true, 1)
	plt.SetColParams("UnCosDiff", false, true, 0, true, 1)

	//plt.SetColParams("Mem", true, true, 0, true, 1)         // default plot
	//plt.SetColParams("TrgOnWasOff", true, true, 0, true, 1) // default plot
	//plt.SetColParams("TrgOffWasOn", true, true, 0, true, 1) // default plot

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", false, true, 0, true, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.TestEnv.Trial.Cur

	row := dt.Rows
	if ss.TestNm == "Train Sat Permutations" && trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("TestNm", row, ss.TestNm)
	dt.SetCellFloat("Trial", row, float64(row))
	dt.SetCellString("TrialName", row, (ss.TestEnv.TrialName.Cur))
	dt.SetCellString("HiddenType", row, ss.HiddenType)
	dt.SetCellString("HiddenFeature", row, (ss.HiddenFeature))
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	//dt.SetCellFloat("Mem", row, ss.Mem)
	//dt.SetCellFloat("TrgOnWasOff", row, ss.TrgOnWasOffCmp)
	//dt.SetCellFloat("TrgOffWasOn", row, ss.TrgOffWasOn)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	// inLay := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	// outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"TestNm", etensor.STRING, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"HiddenType", etensor.STRING, nil, nil},
		{"HiddenFeature", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},

		//{"Mem", etensor.FLOAT64, nil, nil},
		//{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		//{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	// sch = append(sch, etable.Schema{
	// 	{"InAct", etensor.FLOAT64, inLay.Shp.Shp, nil},
	// 	{"OutActM", etensor.FLOAT64, outLay.Shp.Shp, nil},
	// 	{"OutActP", etensor.FLOAT64, outLay.Shp.Shp, nil},
	// }...)
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("TestNm", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("HiddenType", true, true, 0, false, 0)
	plt.SetColParams("HiddenFeature", false, true, 0, false, 0)
	plt.SetColParams("SSE", true, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	//plt.SetColParams("Mem", true, true, 0, true, 1)
	//plt.SetColParams("TrgOnWasOff", true, true, 0, true, 1)
	//plt.SetColParams("TrgOffWasOn", true, true, 0, true, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", false, true, 0, true, .5)
	}

	// plt.SetColParams("InAct", false, true, 0, true, 1)
	// plt.SetColParams("OutActM", false, true, 0, true, 1)
	// plt.SetColParams("OutActP", false, true, 0, true, 1)
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {

	row := dt.Rows
	dt.SetNumRows(row + 1)

	//trl := ss.TstTrlLog
	//tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv        // this is triggered by increment so use previous value
	nt := float64(ss.TestEnv.Trial.Max) // number of trials in view
	shnt := float64(ss.ShTrlNum)
	unnt := float64(ss.UnTrlNum)

	// Computing Epc Shared/Unique feature learning metrics
	ss.EpcShSSE = ss.ShSumSSE / shnt
	ss.ShSumSSE = 0
	ss.EpcShAvgSSE = ss.ShSumAvgSSE / shnt
	ss.ShSumAvgSSE = 0
	ss.EpcShPctErr = float64(ss.ShCntErr) / shnt
	ss.ShCntErr = 0
	ss.EpcShPctCor = 1 - ss.EpcShPctErr
	ss.EpcShCosDiff = ss.ShSumCosDiff / shnt
	ss.ShSumCosDiff = 0
	ss.ShTrlNum = 0

	ss.EpcUnSSE = ss.UnSumSSE / unnt
	ss.UnSumSSE = 0
	ss.EpcUnAvgSSE = ss.UnSumAvgSSE / unnt
	ss.UnSumAvgSSE = 0
	ss.EpcUnPctErr = float64(ss.UnCntErr) / unnt
	ss.UnCntErr = 0
	ss.EpcUnPctCor = 1 - ss.EpcUnPctErr
	ss.EpcUnCosDiff = ss.UnSumCosDiff / unnt
	ss.UnSumCosDiff = 0
	ss.UnTrlNum = 0

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Total Trials", row, float64(nt))
	dt.SetCellFloat("Shared Trials", row, float64(shnt))
	dt.SetCellFloat("ShSSE", row, ss.EpcShSSE)
	dt.SetCellFloat("ShAvgSSE", row, ss.EpcShAvgSSE)
	dt.SetCellFloat("ShPctErr", row, ss.EpcShPctErr)
	dt.SetCellFloat("ShPctCor", row, ss.EpcShPctCor)
	dt.SetCellFloat("ShCosDiff", row, ss.EpcShCosDiff)
	dt.SetCellFloat("Unique Trials", row, float64(unnt))
	dt.SetCellFloat("UnSSE", row, ss.EpcUnSSE)
	dt.SetCellFloat("UnAvgSSE", row, ss.EpcUnAvgSSE)
	dt.SetCellFloat("UnPctErr", row, ss.EpcUnPctErr)
	dt.SetCellFloat("UnPctCor", row, ss.EpcUnPctCor)
	dt.SetCellFloat("UnCosDiff", row, ss.EpcUnCosDiff)

	/*
		trix := etable.NewIdxView(trl)
		spl := split.GroupBy(trix, []string{"TestNm"})
		for _, ts := range ss.TstStatNms {
			split.Agg(spl, ts, agg.AggMean)
		}
		ss.TstStats = spl.AggsToTable(true) // no stat name

		for ri := 0; ri < ss.TstStats.Rows; ri++ {
			tst := ss.TstStats.CellString("TestNm", ri)
			for _, ts := range ss.TstStatNms {
				dt.SetCellFloat(tst+" "+ts, row, ss.TstStats.CellFloat(ts, ri))
			}
		}
	*/

	// base zero on testing performance!
	// DS: Commenting out to test trnlog properly - will get to tstlog later
	//var mem float64
	//if ss.FirstZero < 0 && mem == 1 {
	//	ss.FirstZero = epc
	//}
	//if mem == 1 {
	//	ss.NZero++
	//} else {
	//	ss.NZero = 0
	//}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Total Trials", etensor.INT64, nil, nil},
		{"Shared Trials", etensor.INT64, nil, nil},
		{"ShSSE", etensor.FLOAT64, nil, nil},
		{"ShAvgSSE", etensor.FLOAT64, nil, nil},
		{"ShPctErr", etensor.FLOAT64, nil, nil},
		{"ShPctCor", etensor.FLOAT64, nil, nil},
		{"ShCosDiff", etensor.FLOAT64, nil, nil},
		{"Unique Trials", etensor.INT64, nil, nil},
		{"UnSSE", etensor.FLOAT64, nil, nil},
		{"UnAvgSSE", etensor.FLOAT64, nil, nil},
		{"UnPctErr", etensor.FLOAT64, nil, nil},
		{"UnPctCor", etensor.FLOAT64, nil, nil},
		{"UnCosDiff", etensor.FLOAT64, nil, nil},
	}
	/*for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}*/
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("ShSSE", true, true, 0, false, 0)
	plt.SetColParams("ShAvgSSE", false, true, 0, false, 0)
	plt.SetColParams("ShPctErr", false, true, 0, true, 1)
	plt.SetColParams("ShPctCor", true, true, 0, true, 1)
	plt.SetColParams("ShCosDiff", false, true, 0, true, 1)
	plt.SetColParams("UnSSE", true, true, 0, false, 0)
	plt.SetColParams("UnAvgSSE", false, true, 0, false, 0)
	plt.SetColParams("UnPctErr", false, true, 0, true, 1)
	plt.SetColParams("UnPctCor", true, true, 0, true, 1)
	plt.SetColParams("UnCosDiff", false, true, 0, true, 1)

	/*
		for _, tn := range ss.TstNms {
			for _, ts := range ss.TstStatNms {
				if ts == "Mem" {
					plt.SetColParams(tn+" "+ts, true, true, 0, true, 1) // default plot
				} else {
					plt.SetColParams(tn+" "+ts, false, true, 0, true, 1) // default plot
				}
			}
		}
	*/
	return plt
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" Ge.Avg", cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(ly.Nm+" Act.Avg", cyc, float64(ly.Pools[0].Inhib.Act.Avg))
	}

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " Ge.Avg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + " Act.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, np)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", false, true, 0, false, 0)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" Ge.Avg", true, true, 0, true, .5)
		plt.SetColParams(lnm+" Act.Avg", true, true, 0, true, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TstEpcLog
	epcix := etable.NewIdxView(epclog)
	// compute mean over last N epochs for run level
	nlast := 1
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	params := ss.RunName() // includes tag

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	//dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero)) // DS: Commente out to temporarily get rid of errors
	dt.SetCellFloat("ShSSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	/*
		for _, tn := range ss.TstNms {
			for _, ts := range ss.TstStatNms {
				nm := tn + " " + ts
				dt.SetCellFloat(nm, row, agg.Mean(epcix, nm)[0])
			}
		}
	*/

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	for _, tn := range ss.TstNms {
		nm := tn + " " + "Mem"
		split.Desc(spl, nm)
	}
	split.Desc(spl, "FirstZero")
	ss.RunStats = spl.AggsToTable(false)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	//if ss.RunFile != nil {
	//	if row == 0 {
	//		dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
	//	}
	//	dt.WriteCSVRow(ss.RunFile, row, etable.Tab, true)
	//}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}

	/*
		for _, tn := range ss.TstNms {
			for _, ts := range ss.TstStatNms {
				sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
			}
		}
	*/
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("FirstZero", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	/*
		for _, tn := range ss.TstNms {
			for _, ts := range ss.TstStatNms {
				if ts == "Mem" {
					plt.SetColParams(tn+" "+ts, true, true, 0, true, 1) // default plot
				} else {
					plt.SetColParams(tn+" "+ts, false, true, 0, true, 1)
				}
			}
		}
	*/
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("slp-rep")
	gi.SetAppAbout(`This is a Sleep-replay model developed in Leabra. See <a href="https://github.com/schapirolab/sleep-replay"> on GitHub</a>.</p>`)

	win := gi.NewMainWindow("slp-rep", "Sleep-replay", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html
	nv.SetNet(ss.Net)
	ss.NetView = nv
	nv.ViewDefaults()

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "SlpCycPlot").(*eplot.Plot2D)
	ss.SlpCycPlot = ss.ConfigSlpCycPlot(plt, ss.SlpCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't return on trial -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, true, true) // contains, ignoreCase
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, true, false, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.MaxRuns, "runs", 30, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", false, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc" + strconv.Itoa(int(ss.RndSeed)))
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs\n", ss.MaxRuns)
	ss.Train()
}
